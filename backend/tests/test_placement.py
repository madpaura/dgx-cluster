"""Placement engine: does it fit, where is best, and why not there."""
from __future__ import annotations

from app.models import Deployment, DeployStatus, Gpu, Node, NodeStatus
from app.services.placement import RESERVE_MB, busy_gpu_indices, plan

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


def occupy(node, indices, status=DeployStatus.healthy):
    dep = Deployment(
        id=f"dep-{node.name}-{indices}", served_model_name="x", hf_repo="x/y",
        node_id=node.id, gpu_indices=list(indices), port=8100, status=status,
        tensor_parallel_size=len(indices),
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


def test_skips_gpus_already_claimed_by_an_active_deployment():
    node = make_node("dgx-01", H100, 8)
    occupy(node, [0, 1])
    options, _ = plan([node], per_gpu_gb=40, tp=2)
    assert options[0].gpu_indices == [2, 3]


def test_a_failed_deployment_releases_its_gpus():
    node = make_node("dgx-01", H100, 8)
    occupy(node, [0, 1], status=DeployStatus.failed)
    assert busy_gpu_indices(node) == set()
    options, _ = plan([node], per_gpu_gb=40, tp=2)
    assert options[0].gpu_indices == [0, 1]


def test_prefers_an_aligned_nvlink_group():
    node = make_node("dgx-01", H100, 8)
    occupy(node, [0])                      # leaves 1..7 free
    options, _ = plan([node], per_gpu_gb=40, tp=4)
    assert options[0].gpu_indices == [4, 5, 6, 7]
    assert options[0].note == "aligned NVLink group"


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
    assert "needs 71 GiB per GPU" in rejections[0].reason
    assert "largest free GPU has 48 GiB" in rejections[0].reason


def test_rejection_counts_free_gpus_not_total():
    node = make_node("dgx-01", H100, 8)
    occupy(node, [0, 1, 2, 3, 4, 5, 6])
    options, rejections = plan([node], per_gpu_gb=40, tp=4)
    assert not options
    assert rejections[0].reason == "only 1 of 8 GPUs free, needs 4"


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
    """A model that exactly equals free VRAM must not be placed: CUDA context
    and fragmentation need room."""
    exact = make_node("tight", RTX, 1, vram_mb=40 * 1024)
    options, rejections = plan([exact], per_gpu_gb=40, tp=1)
    assert not options
    roomy = make_node("roomy", RTX, 1, vram_mb=40 * 1024 + RESERVE_MB)
    options, _ = plan([roomy], per_gpu_gb=40, tp=1)
    assert options


def test_excluded_nodes_are_skipped_silently():
    nodes = [make_node("a", H100, 8), make_node("b", H100, 8)]
    options, _ = plan(nodes, per_gpu_gb=40, tp=1, exclude_node_ids={"a"})
    assert [o.node_name for o in options] == ["b"]
