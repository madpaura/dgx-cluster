"""Placement engine: does it fit, where is best, and why not there."""
from __future__ import annotations

from app.models import Deployment, DeployStatus, Gpu, Node, NodeStatus
from app.services.capacity import RESERVE_MB, reserved_mb
from app.services.placement import plan

H100 = "NVIDIA H100 80GB HBM3"
A100 = "NVIDIA A100-SXM4-80GB"
RTX = "NVIDIA RTX 6000 Ada Generation"


def make_node(name, gpu_model, count, vram_mb=81559, status=NodeStatus.online, used_mb=0, labels=None):
    node = Node(id=name, name=name, hostname=f"{name}.local", status=status, labels=labels or {})
    node.gpus = [
        Gpu(id=f"{name}-{i}", node_id=name, index=i, name=gpu_model,
            memory_total_mb=vram_mb, memory_used_mb=used_mb)
        for i in range(count)
    ]
    node.deployments = []
    return node


def occupy(node, indices, status=DeployStatus.healthy, reserved_gb=None):
    """Place a model on some GPUs. `reserved_gb` is what it holds on each; the
    default takes the whole card, which is what an exclusive model does."""
    per_gpu = node.gpus[0].memory_total_mb if reserved_gb is None else int(reserved_gb * 1024)
    dep = Deployment(
        id=f"dep-{node.name}-{indices}", served_model_name="x", hf_repo="x/y",
        node_id=node.id, gpu_indices=list(indices), port=8100, status=status,
        tensor_parallel_size=len(indices), reserved_mb_per_gpu=per_gpu,
    )
    node.deployments.append(dep)
    return dep


def test_finds_a_home_on_an_empty_fleet():
    nodes = [make_node("dgx-01", H100, 8)]
    options, rejections = plan(nodes, per_gpu_gb=40, tp=2)
    assert [o.node_name for o in options] == ["dgx-01"]
    assert options[0].gpu_indices == [0, 1]
    assert not rejections


def test_best_fit_prefers_the_node_with_least_spare_capacity():
    """A small model should land on a workstation rather than fragmenting a DGX,
    so large contiguous blocks stay available for models that need them."""
    nodes = [make_node("dgx-01", H100, 8), make_node("rtx-ws-01", RTX, 2, vram_mb=49140)]
    options, _ = plan(nodes, per_gpu_gb=20, tp=1)
    assert options[0].node_name == "rtx-ws-01"
    assert options[-1].node_name == "dgx-01"


def test_skips_gpus_whose_remaining_vram_is_too_small():
    node = make_node("dgx-01", H100, 8)
    occupy(node, [0, 1])                       # takes both cards whole
    options, _ = plan([node], per_gpu_gb=40, tp=2)
    assert options[0].gpu_indices == [2, 3]


def test_a_second_model_shares_a_gpu_that_still_has_room():
    """An 8B model on an 80 GB card leaves most of it idle; the next small
    model belongs there, not on a fresh card."""
    node = make_node("dgx-01", H100, 8)
    occupy(node, [0], reserved_gb=22)
    options, _ = plan([node], per_gpu_gb=22, tp=1)
    assert options[0].gpu_indices == [0], "should pack onto the partly used card"
    assert options[0].shares_with == 1
    assert "shares with 1 model" in options[0].note


def test_sharing_stops_when_the_remaining_vram_runs_out():
    node = make_node("dgx-01", H100, 1)
    occupy(node, [0], reserved_gb=50)
    options, rejections = plan([node], per_gpu_gb=40, tp=1)
    assert not options
    assert "1 of 1 GPUs have that" not in rejections[0].reason
    assert "GiB free per GPU" in rejections[0].reason


def test_three_small_models_fit_on_one_card():
    node = make_node("dgx-01", H100, 1)
    occupy(node, [0], reserved_gb=20)
    node.deployments[-1].id = "a"
    occupy(node, [0], reserved_gb=20)
    node.deployments[-1].id = "b"
    options, _ = plan([node], per_gpu_gb=20, tp=1)
    assert options and options[0].gpu_indices == [0]
    assert options[0].shares_with == 2


def test_packing_is_preferred_over_opening_another_card():
    """Keeping whole GPUs free is what lets a big model land later. Offered one
    node with a partly used card and three empty ones, the small model belongs
    on the used one."""
    node = make_node("dgx-01", H100, 4)
    occupy(node, [0], reserved_gb=20)
    options, _ = plan([node], per_gpu_gb=20, tp=1)
    assert options[0].gpu_indices == [0]

    untouched = make_node("dgx-02", H100, 4)
    ranked, _ = plan([node, untouched], per_gpu_gb=20, tp=1)
    assert ranked[0].node_name == "dgx-01", "pack before opening a second box"


def test_a_failed_deployment_releases_its_vram():
    node = make_node("dgx-01", H100, 8)
    occupy(node, [0, 1], status=DeployStatus.failed)
    assert reserved_mb(node) == {}
    options, _ = plan([node], per_gpu_gb=40, tp=2)
    assert options[0].gpu_indices == [0, 1]


def test_prefers_an_aligned_nvlink_group():
    node = make_node("dgx-01", H100, 8)
    occupy(node, [0])                      # leaves 1..7 with room
    options, _ = plan([node], per_gpu_gb=40, tp=4)
    assert options[0].gpu_indices == [4, 5, 6, 7]
    assert "aligned NVLink group" in options[0].note


def test_never_groups_unlike_gpus():
    node = make_node("mixed", H100, 2)
    node.gpus[1].name = A100
    options, rejections = plan([node], per_gpu_gb=40, tp=2)
    assert not options
    assert "homogeneous" in rejections[0].reason


def test_rejection_says_how_many_gpus_are_short():
    nodes = [make_node("rtx-ws-01", RTX, 2, vram_mb=49140)]
    options, rejections = plan(nodes, per_gpu_gb=40, tp=4)
    assert not options
    assert rejections[0].reason == "has 2 GPUs, needs 4"


def test_rejection_says_how_much_vram_is_short():
    nodes = [make_node("rtx-ws-01", RTX, 2, vram_mb=49140)]
    options, rejections = plan(nodes, per_gpu_gb=70, tp=2)
    assert not options
    assert "needs 70 GiB free per GPU" in rejections[0].reason
    assert "0 of 2 GPUs have that" in rejections[0].reason
    assert "most free: 47 GiB" in rejections[0].reason


def test_rejection_counts_gpus_with_room_not_gpus_that_are_empty():
    node = make_node("dgx-01", H100, 8)
    occupy(node, [0, 1, 2, 3, 4, 5, 6])
    options, rejections = plan([node], per_gpu_gb=40, tp=4)
    assert not options
    assert "1 of 8 GPUs have that" in rejections[0].reason


def test_unreachable_and_draining_nodes_are_excluded_with_a_reason():
    nodes = [
        make_node("down", H100, 8, status=NodeStatus.unreachable),
        make_node("draining", H100, 8, status=NodeStatus.draining),
        make_node("maint", H100, 8, status=NodeStatus.maintenance),
    ]
    options, rejections = plan(nodes, per_gpu_gb=40, tp=1)
    assert not options
    assert {r.reason for r in rejections} == {
        "node is unreachable", "node is draining", "node is maintenance",
    }


def test_required_labels_filter_the_fleet():
    nodes = [
        make_node("ib", H100, 8, labels={"net": "ib"}),
        make_node("eth", H100, 8, labels={"net": "eth"}),
    ]
    options, rejections = plan(nodes, per_gpu_gb=40, tp=1, required_labels={"net": "ib"})
    assert [o.node_name for o in options] == ["ib"]
    assert rejections[0].node_name == "eth"


def test_headroom_is_reserved_on_top_of_the_model_size():
    """A model that exactly equals the card must not be placed: CUDA context
    and fragmentation need room."""
    exact = make_node("tight", RTX, 1, vram_mb=40 * 1024)
    options, _ = plan([exact], per_gpu_gb=40, tp=1)
    assert not options
    roomy = make_node("roomy", RTX, 1, vram_mb=40 * 1024 + RESERVE_MB)
    options, _ = plan([roomy], per_gpu_gb=40, tp=1)
    assert options


def test_excluded_nodes_are_skipped_silently():
    nodes = [make_node("a", H100, 8), make_node("b", H100, 8)]
    options, _ = plan(nodes, per_gpu_gb=40, tp=1, exclude_node_ids={"a"})
    assert [o.node_name for o in options] == ["b"]
