import tensorflow as tf

class MCCMetric(tf.keras.metrics.Metric):
    """Computes Matthews Correlation Coefficient

    Defined as (tp*tn)-(fp*fn)/sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))"""
    def __init__(self, name=None, dtype=None):
        super(MCCMetric, self).__init__(name=name, dtype=dtype)
        self.tp_ = tf.keras.metrics.TruePositives()
        self.tn_ = tf.keras.metrics.TrueNegatives()
        self.fp_ = tf.keras.metrics.FalsePositives()
        self.fn_ = tf.keras.metrics.FalseNegatives()
    
    def reset_states(self):
        self.tp_.reset_states()
        self.tn_.reset_states()
        self.fp_.reset_states()
        self.fn_.reset_states()
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.tp_.update_state(y_true[:,:,:,1], y_pred[:,:,:,1], sample_weight)
        self.tn_.update_state(y_true[:,:,:,1], y_pred[:,:,:,1], sample_weight)
        self.fp_.update_state(y_true[:,:,:,1], y_pred[:,:,:,1], sample_weight)
        self.fn_.update_state(y_true[:,:,:,1], y_pred[:,:,:,1], sample_weight)
    
    def result(self):
        tp = self.tp_.result()
        tn = self.tn_.result()
        fp = self.fp_.result()
        fn = self.fn_.result()
        return ((tp*tn)-(fp*fn))/((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**(.5)