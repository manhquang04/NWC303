| rq | task | reward | agent | precision | recall | f1 | fpr | auroc | pr_auc | claim |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RQ2 | Rogue AP | aggressive_reward | dqn | 0.4868 | 0.8237 | 0.6119 | 0.4342 | 0.7506 | 0.6082 | reward controls recall/FPR trade-off |
| RQ2 | Rogue AP | aggressive_reward | dueling_dqn | 0.4705 | 0.8192 | 0.5977 | 0.4609 | 0.7182 | 0.5171 | reward controls recall/FPR trade-off |
| RQ2 | Rogue AP | aggressive_reward | double_dqn | 0.4852 | 0.7299 | 0.5829 | 0.3873 | 0.7403 | 0.5903 | reward controls recall/FPR trade-off |
| RQ2 | Rogue AP | conservative_reward | dueling_dqn | 0.4790 | 0.7388 | 0.5812 | 0.4018 | 0.7261 | 0.5627 | reward controls recall/FPR trade-off |
| RQ2 | Rogue AP | conservative_reward | double_dqn | 0.4834 | 0.6518 | 0.5551 | 0.3482 | 0.7211 | 0.5556 | reward controls recall/FPR trade-off |
| RQ2 | ARP Spoofing | conservative_reward | DQN window | 0.6829 | 0.9551 | 0.7964 | 0.6675 | 0.7665 | 0.7757 | offline reward ablation; not stronger than SDN runtime policy |
| RQ2 | ARP Spoofing | aggressive_reward | DQN window | 0.6094 | 0.9668 | 0.7476 | 0.9325 | 0.7412 | 0.8323 | offline reward ablation; not stronger than SDN runtime policy |
| RQ2 | ARP Spoofing | recall_prioritized_reward | DQN window | 0.5976 | 0.9867 | 0.7444 | 1.0000 | 0.5415 | 0.6427 | offline reward ablation; not stronger than SDN runtime policy |
| RQ2 | ARP Spoofing | fpr_constrained_reward | DQN window | 0.5918 | 0.9535 | 0.7303 | 0.9900 | 0.5368 | 0.6759 | offline reward ablation; not stronger than SDN runtime policy |
