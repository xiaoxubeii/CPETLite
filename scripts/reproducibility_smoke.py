"""Patient-free checks of the frozen source's teacher/package loss contract."""
import json
import torch
from cpet_foundation.model.losses import total_loss
from cpet_foundation.model.wearable_student import MEASURED_TARGET_LOSS_KEYS,wearable_student_loss

def main():
    mu=torch.zeros(2,8,3,requires_grad=True)
    latent=torch.zeros(2,8,4,requires_grad=True)
    outputs={'trajectory_mu':mu,'trajectory_logvar':torch.zeros_like(mu),
             'latent_rollout':latent,'at_norm':torch.full((2,),0.5), 'vo2peak':torch.full((2,),20.)}
    mask=torch.ones(2,8);mask[:,-1]=0
    teacher_batch={'x':mu.detach().clone(),'mask':mask,'feature_mask':torch.ones_like(mu),
                   'at_norm':torch.full((2,),0.5),'at_supervision_mask':torch.tensor([1.,0.]),
                   'vo2peak':torch.full((2,),20.),'vo2peak_mask':torch.ones(2)}
    teacher,_=total_loss(outputs,teacher_batch,{'VO2':0,'VCO2':1,'VE':2})
    package={'mask':mask,'teacher_trajectory_mu':mu.detach().clone(),
             'teacher_trajectory_logvar':torch.zeros_like(mu),'teacher_latent_rollout':latent.detach().clone(),
             'teacher_at_norm':torch.full((2,),0.5),'teacher_at_mask':torch.tensor([1.,0.]),
             'teacher_vo2peak':torch.full((2,),20.),'teacher_vo2peak_mask':torch.ones(2)}
    weights={k:0. for k in MEASURED_TARGET_LOSS_KEYS}
    weights.update(latent_alignment=0.75,trajectory_distillation=0.75,uncertainty_distillation=0.2,
                   endpoint=1.,physiology_consistency=0.1,future_window=0.)
    student,_=wearable_student_loss(outputs,package,weights,supervision_mode='teacher_only')
    assert torch.isfinite(teacher)and torch.isfinite(student)
    assert abs(float(teacher.detach()))<1e-7 and abs(float(student.detach()))<1e-7
    (teacher+student).backward();assert mu.grad is not None
    contaminated={**package,'target_trajectory':mu.detach()}
    try:wearable_student_loss(outputs,contaminated,weights,supervision_mode='teacher_only')
    except ValueError:reject=True
    else:reject=False
    assert reject
    print(json.dumps({'passed':True,'teacher_loss':float(teacher.detach()),'package_loss':float(student.detach()),
                      'measured_target_rejected':reject,'boundary':'Synthetic loss/gradient/access-contract smoke only; not clinical metric reproduction.'}))

if __name__=='__main__':main()
