from argparse import ArgumentParser


def add_bool_argument(arg_parser, true_arg, false_arg, dest_var, help_str):
    arg_parser.add_argument(true_arg, dest=dest_var, action='store_true', help=help_str)
    arg_parser.add_argument(false_arg, dest=dest_var, action='store_false', help=help_str)


def make_args():
    parser = ArgumentParser()
    parser.add_argument('--cuda', dest='cuda', default='0', type=str)  # -1 if not using GPU
    parser.add_argument('--comment', dest='comment', default='0', type=str, help='comment')

    # Data related
    parser.add_argument('--batch_size', default=8, type=int, help='size of graph batching')
    # add_bool_argument(parser, "--if_curriculum", "--if_curriculum_no", dest_var="if_curriculum", help_str="if use curriculum")
    parser.add_argument('--train_data_dir', default='Data/6types-processed_data', type=str, help='where to load the training data if not preload')
    parser.add_argument('--raw_dir', default='Data/6types-raw_data', type=str, help='For evaluation to copy and paste')  # normal runs
    # parser.add_argument('--preload_dir', default='Data/6types_preload_data.pkl', type=str, help='where to load the training data')

    # 96000, 96005, 96003
    parser.add_argument('--train_size', default=96000, type=int, help='how many data are train data')  #
    parser.add_argument('--test_size', default=4000, type=int, help='how many data are test data')  #
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")  # 8, if RAM OOM -> decrease to 4
    parser.add_argument("--variation_eval_id1", type=int, default=96018, help="Data index for variation test")  # 96008
    parser.add_argument("--variation_eval_id2", type=int, default=96010, help="Data index for variation test")
    parser.add_argument("--variation_num", type=int, default=25, help="How many variation to generate for the variation test data")

    # -------------------------------------------------------------------------------------------------

    # Model related
    parser.add_argument("--latent_dim", type=int, default=128, help="dimensionality of the latent space")
    parser.add_argument("--noise_dim", type=int, default=32, help="dimensionality of the noise space")
    parser.add_argument("--program_layer", type=int, default=4, help="numbers of message passing in program graph")
    parser.add_argument("--voxel_layer", type=int, default=12, help="numbers of message passing in voxel graph")

    # Loss related
    parser.add_argument('--gan_loss', default='WGANGP', type=str, help='WGANGP/NSGAN/LSGAN/hinge')
    parser.add_argument("--gp_lambda", type=float, default=10.0, help="gradient penalty coefficient in D loss")
    parser.add_argument("--lp_weight", type=float, default=0.0, help="link prediction coefficient in G loss")
    parser.add_argument("--tr_weight", type=float, default=0.0, help="program target ratio coefficient in G loss")
    parser.add_argument("--far_weight", type=float, default=0.0, help="far coefficient in G loss")

    # Link prediction related
    # link prediction is not used in this implementation
    parser.add_argument("--lp_sample_size", type=int, default=20, help="link prediction sample size")
    parser.add_argument('--lp_similarity_fun', default='cos', type=str, help='link prediction similarity type: cos/dot/l2/mlp')
    parser.add_argument('--lp_loss_fun', default='hinge', type=str, help='link prediction loss type: hinge/BCE/skipgram')
    parser.add_argument("--lp_hinge_margin", type=float, default=1.0, help="link prediction hinge loss margin")

    # Training parameter
    parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
    parser.add_argument("--n_critic_d", type=int, default=1, help="number of training steps for discriminator per iter")
    parser.add_argument("--n_critic_g", type=int, default=5, help="number of training steps for discriminator per iter")
    parser.add_argument("--n_critic_p", type=int, default=5, help="number of training steps for discriminator per iter")
    parser.add_argument("--plot_period", type=int, default=10, help="number of epochs between each result plot")  # 10
    parser.add_argument("--eval_period", type=int, default=20, help="number of epochs between each evaluation and save model")  # 20

    # Optimizer parameter
    parser.add_argument("--g_lr", type=float, default=0.0001, help="adam: learning rate")
    parser.add_argument("--d_lr", type=float, default=0.0001, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")

    parser.set_defaults(if_curriculum=False)
    args = parser.parse_args()
    return args
