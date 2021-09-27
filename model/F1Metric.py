import tensorflow as tf

class F1Metric(tf.keras.metrics.Metric):
    def __init__(self, name=None, dtype=None):
        super(F1Metric, self).__init__(name=name, dtype=dtype)
        self.tp_ = tf.keras.metrics.TruePositives()
        self.fp_ = tf.keras.metrics.FalsePositives()
        self.fn_ = tf.keras.metrics.FalseNegatives()
    
    def reset_states(self):
        self.tp_.reset_states()
        self.fp_.reset_states()
        self.fn_.reset_states()
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.tp_.update_state(y_true[:,:,:,1], y_pred[:,:,:,1], sample_weight)
        self.fp_.update_state(y_true[:,:,:,1], y_pred[:,:,:,1], sample_weight)
        self.fn_.update_state(y_true[:,:,:,1], y_pred[:,:,:,1], sample_weight)
    
    def result(self):
        tp = self.tp_.result()
        fp = self.fp_.result()
        fn = self.fn_.result()
        return (2*tp)/(2*tp+fp+fn)