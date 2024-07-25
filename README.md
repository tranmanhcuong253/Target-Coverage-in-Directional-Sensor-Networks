# Target Coverage in Directional Sensor Networks
Maximum target coverage by adjusting the orientation of distributed sensors in directional sensor networks using Deep Reinforcement Learning
# Training
To train the orginal model
```
python main.py --env Pose-v1 --model multi-att-shap --workers 6
```
To train my new model 
```
python main.py --env Pose-v1 --model fm-att-shap --workers 6
```

# Reference
```bibtex
@article{xu2020learning,
  title={Learning Multi-Agent Coordination for Enhancing Target Coverage in Directional Sensor Networks},
  author={Xu, Jing and Zhong, Fangwei and Wang, Yizhou},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}


