Qualitative segmentation export (11 per folder)
================================================================

  bijie_landslide/      — Bijie landslide/ (GT + pred)
  bijie_non_landslide/  — Bijie non-landslide/ (empty GT; pred should be empty)
  landslide4sense/      — L4S TrainData (GT + pred)

Files per stem: {stem}_image.png, {stem}_mask_gt.png, {stem}_mask_pred.png

Checkpoints:
  Bijie: /home/user/Desktop/Deep_learning_projects/CSIR/Geology-Aware_Hybrid_SAM/runs/bijie/tri_encoder_cfm_v2/checkpoint/best.pt (thr=0.6)
  L4S:   /home/user/Desktop/Deep_learning_projects/CSIR/Geology-Aware_Hybrid_SAM/runs/landslide4sense/tri_encoder_cfm_v2/checkpoint/best.pt (thr=0.6)

Stems:
  bijie_landslide: df027, hz070, hz080, js006, ny017, ny039, ny070, zj007, zj045, zj107, zj112
  bijie_non_landslide: dhzgf10144, dzjwv05294, fyb1254, fyb1409, fyb1580, fyb436, fyb481, fyb486, fyb603, fyb66, fyb967
  landslide4sense: 1105, 1428, 165, 1829, 2066, 2398, 2504, 2916, 2997, 3585, 757
