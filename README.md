<p align="center">

  <h1 align="center">COPlanner: Plan to Roll Out Conservatively but to Explore Optimistically for Model-Based RL</h1>
  <h2 align="center">ICLR 2024 <a href="https://openreview.net/forum?id=jnFcKjtUPN">Poster</a></h2>
  <p align="center">
    <a><strong>Xiyao Wang</strong></a>
    ·
    <a><strong>Ruijie Zheng</strong></a>
    ·
    <a><strong>Yanchao Sun</strong></a>
    ·
    <a><strong>Ruonan Jia</strong></a>
    ·
    <a><strong>Wichayaporn Wongkamjan</strong></a>
    ·
    <a><strong>Huazhe Xu</strong></a>
    ·
    <a><strong>Furong Huang</strong></a>
  </p>

</p>

<h3 align="center">
  <a href="https://arxiv.org/abs/2310.07220"><strong>arXiv</strong></a>
</h3>

<div align="center">
  <img src="./doc/coplanner_framework.png" alt="Logo" width="100%">
</div>

# 🛠️ Usage
We provide scripts to train and evaluate policies of different backbones (MBPO, DreamerV2, and DreamerV3) in separate folders. 

For MBPO, in the MuJoCo environment, we implement COplanner based on [mbpo_pytorch](https://github.com/Xingyu-Lin/mbpo_pytorch). In DeepMind Control Suite, we utilize [MBRL-lib](https://github.com/facebookresearch/mbrl-lib).

# 🏷️ License
This repository is released under the MIT license. See [LICENSE](LICENSE) for additional details.

# 🙏 Acknowledgement

Our code is built upon [mbpo_pytorch](https://github.com/Xingyu-Lin/mbpo_pytorch), [MBRL-lib](https://github.com/facebookresearch/mbrl-lib), [DreamerV2-pytorch](https://github.com/jsikyoon/dreamer-torch), and [DreamerV3](https://github.com/danijar/dreamerv3). We thank all these authors for their nicely open sourced code and their great contributions to the community.

# 📝 Citation

If you find our work useful, please consider citing:
```
@article{wang2023coplanner,
  title={COPlanner: Plan to Roll Out Conservatively but to Explore Optimistically for Model-Based RL},
  author={Wang, Xiyao and Zheng, Ruijie and Sun, Yanchao and Jia, Ruonan and Wongkamjan, Wichayaporn and Xu, Huazhe and Huang, Furong},
  journal={arXiv preprint arXiv:2310.07220},
  year={2023}
}
```
