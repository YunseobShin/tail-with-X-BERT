class Hyperparameters():
    def __init__(self, dataset):
        self.max_seq_len = 256
        if dataset == 'Eurlex-4K':
        	self.dataset = 'Eurlex-4K'
        	self.depth=6
        	self.train_batch_size=8
        	self.eval_batch_size=48
        	self.log_interval=100
        	self.eval_interval=400
        	self.learning_rate=1e-4
        	self.warmup_rate=0.1
        elif dataset == 'Wiki10-31K':
            self.dataset = 'Wiki10-31K'
            self.depth = 9
            self.train_batch_size=8
            self.eval_batch_size=16
            self.log_interval=200
            self.eval_interval=150
            self.learning_rate=5e-5
            self.warmup_rate=0.1
            # self.max_seq_len = 512
        elif dataset == 'AmazonCat-13K':
            self.dataset = 'AmazonCat-13K'
            self.depth=8
            self.train_batch_size=8
            self.eval_batch_size=48
            self.log_interval=2000
            self.eval_interval=100
            self.learning_rate=5e-5
            self.warmup_rate=0.1
        elif dataset == 'Wiki-500K':
            self.dataset = 'Wiki-500K'
            self.depth=13
            self.train_batch_size=12
            self.eval_batch_size=36
            self.log_interval=1000
            self.eval_interval=15000
            self.learning_rate=8e-5
            self.warmup_rate=0.2
        else:
            print("unknown dataset for the experiment!")
            exit()
