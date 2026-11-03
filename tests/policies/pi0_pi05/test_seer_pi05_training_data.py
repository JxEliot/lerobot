import numpy as np
import torch
from scipy.spatial.transform import Rotation
from lerobot.datasets.dataset_reader import DatasetReader
from lerobot.datasets.seer_pi05_stats import relative_action_windows
from lerobot.policies.pi05.processor_pi05 import _relative_eef_pose


def test_quaternion_conversion_sign_and_scipy():
    rng=np.random.default_rng(42)
    q=Rotation.random(2000, random_state=rng).as_quat()
    q=np.concatenate((q, -q, [[0,0,0,1], [0,0,0,-1], [1,0,0,0]]))
    poses=torch.tensor(np.c_[np.zeros((len(q),3)),q],dtype=torch.float64)
    actual=DatasetReader._quat_to_rotvec(poses).numpy()[:,3:]
    error=(Rotation.from_rotvec(actual).inv()*Rotation.from_quat(q)).magnitude()
    assert error.max()<1e-7
    np.testing.assert_allclose(actual[:2000],actual[2000:4000],atol=1e-7)


def test_windows_match_training_processor_and_boundary():
    rng=np.random.default_rng(4)
    state=rng.normal(size=(4,30)).astype(np.float32)*.1
    action=rng.normal(size=(4,14)).astype(np.float32)*.1
    actual=relative_action_windows(state,action,5)
    indices=np.minimum(np.arange(4)[:,None]+np.arange(5),3)
    expected=action[indices].copy()
    for a0,s0 in [(0,18),(6,24)]:
        expected[...,a0:a0+6]=_relative_eef_pose(torch.tensor(expected[...,a0:a0+6]),torch.tensor(state[:,None,s0:s0+6])).numpy()
    np.testing.assert_allclose(actual,expected,atol=2e-6)
    np.testing.assert_array_equal(actual[:,:,12:],action[indices,:,][...,12:])


def test_relative_absolute_roundtrip_near_pi():
    from lerobot.policies.pi05.processor_pi05 import _absolute_eef_pose, _matrix_to_rotvec
    rng=np.random.default_rng(71)
    axes=rng.normal(size=(2000,3)); axes/=np.linalg.norm(axes,axis=1,keepdims=True)
    angles=np.r_[np.full(500,np.pi), np.full(500,np.pi-1e-5), np.full(500,np.pi+1e-5), np.full(500,1e-9)]
    rotations=Rotation.from_rotvec(axes*angles[:,None])
    actual=_matrix_to_rotvec(torch.tensor(rotations.as_matrix(),dtype=torch.float32)).numpy()
    assert (Rotation.from_rotvec(actual).inv()*rotations).magnitude().max()<2e-6
    target=np.c_[rng.normal(size=(2000,3)),rotations.as_rotvec()]
    current=np.c_[rng.normal(size=(2000,3)),Rotation.random(2000,random_state=rng).as_rotvec()]
    t=torch.tensor(target,dtype=torch.float32); c=torch.tensor(current,dtype=torch.float32)
    result=_absolute_eef_pose(_relative_eef_pose(t,c),c).numpy()
    np.testing.assert_allclose(result[:,:3],target[:,:3],atol=3e-6)
    assert (Rotation.from_rotvec(result[:,3:]).inv()*rotations).magnitude().max()<3e-6
