import tensorflow as tf

class CustomExponentialDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, decay_factor, decay_steps, end_rate):
        self.initial_learning_rate = initial_learning_rate
        self.decay_factor = decay_factor
        self.decay_steps = decay_steps
        self.end_rate = end_rate

    def __call__(self, step):
        lr = self.initial_learning_rate * self.decay_factor ** (step // self.decay_steps)
        lr = tf.maximum(lr, self.end_rate)
        return lr

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_factor": self.decay_factor,
            "decay_steps": self.decay_steps,
            "end_rate": self.end_rate
        }