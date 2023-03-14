from visdom import Visdom


visdom_file = "../data/visdom.log"


viz = Visdom(env="demo", log_to_filename=visdom_file)
Visdom.replay_log(viz, log_filename=visdom_file)