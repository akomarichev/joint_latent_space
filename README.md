## Learning Geometry-Aware Joint Latent Space for Simultaneous Multimodal Shape Generation
Created by <a href="https://github.com/artemkomarichev" target="_blank">Artem Komarichev</a>, <a href="http://www.cs.wayne.edu/~jinghua/" target="_blank">Jing Hua</a>, <a href="http://www.cs.wayne.edu/zzhong/" target="_blank">Zichun Zhong</a> from Department of Computer Science, Wayne State University.

![teaser image](https://github.com/artemkomarichev/joint_latent_space/blob/main/pics/teaser.png)

### Introduction

Our paper (<a href="https://zichunzhong.github.io/papers/JointLatent_CAGD2022.pdf" target="_blank">paper</a>, <a href="https://zichunzhong.github.io/papers/JointLatent_Supp_CAGD2022_LR.pdf" target="_blank">supplementary</a>) proposes a new approach to learn geometry-aware joint latent space.

To appear, Computer Aided Geometric Design (GMP 2022), May 2022

We provide the code of our models that was tested with Tensorflow 1.13.1, CUDA 10.0, and python 3.7 on Ubuntu 16.04. We run all our experiments on a single NVIDIA Titan Xp GPU with 12GB GDDR5X.

### Data

<a href="https://waynestateprod-my.sharepoint.com/:u:/g/personal/fy7555_wayne_edu/ES0LnFwtCMhDlm1czqp3G1cBp5F58Dk1Nr7dyqwaSZm0Qg?e=kXPce9">Download</a> our prepared *ShapeNet Core* dataset first. Point clouds are sampled from meshes with 10K points (XYZ + normals) per shape with its rendered images.

### Geometry-Aware Autoencoder (GAE)

The architecture of our proposed GAE on point clouds is shown below:

![gae image](https://github.com/artemkomarichev/joint_latent_space/blob/main/pics/gae.png)
    
  To train GAE model on *ShapeNet Core* dataset type the following command:

        python main_ae_pc_adaptive.py --to_train=1 --log_dir=output/cyclegan/exp_ae_pc --config_filename=configs/exp_ae_pc.json

  To evaluate a trained model run the following script:

        python main_ae_pc_adaptive.py --to_train=0 --checkpoint_dir=output/cyclegan/exp_ae_pc/20220213-...... --log_dir=output/cyclegan/exp_ae_pc --config_filename=configs/exp_ae_pc.json

### Autoencoder on images
    
  To train autoencoder on images on *ShapeNet Core* dataset type the following command:

        python main_ae_img.py --to_train=1 --log_dir=output/cyclegan/exp_ae_img --config_filename=configs/exp_ae_img.json

  To evaluate a trained model run the following script:

        python main_ae_img.py --to_train=0 --checkpoint_dir=output/cyclegan/exp_ae_img/20220213-...... --log_dir=output/cyclegan/exp_ae_img --config_filename=configs/exp_ae_img.json

### Mixer and Joint Generative Model (Coming Soon!)

### Citation
If you find our work useful in your research, please cite our work:

    @article{komarichev2022learning,
        title={Learning geometry-aware joint latent space for simultaneous multimodal shape generation},
        author={Komarichev, Artem and Hua, Jing and Zhong, Zichun},
        journal={Computer Aided Geometric Design},
        volume={93},
        pages={102076},
        year={2022},
        publisher={Elsevier}
    }