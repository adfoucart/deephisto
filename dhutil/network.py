import tensorflow as tf

'''
Initialize all uninitialized variables in the session.
'''
def initialize_uninitialized(sess):
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))

'''
Train a network on a dataset

An "epoch" is defined here as going once through all images in the dataset 
(note that this doesn't mean seeing all the *patches* in the dataset. Every epoch will see different patches!)
It's therefore normal here to have "a lot" of epochs (>1000), because an "epoch" actually sees only a small part of the dataset.
'''
def train(net, feed, nPatchesInValSet=40, batch_size=20, epochs=1500, threaded=False):
    trainingStep,loss,acc = net.train()

    # Initializing everything
    with net.mainGraph.as_default():
        if( net.trainAE == True ):
            tf.summary.scalar('aeloss', loss)
        else:
            tf.summary.scalar('clfloss', loss)
            tf.summary.scalar('accuracy', acc)

        merged = tf.summary.merge_all()
        saver = tf.train.Saver()
        train_writer = tf.summary.FileWriter('%s/%s'%(net.summaries_dir, net.clf_name), net.sess.graph)
        
        initialize_uninitialized(net.sess)
    if net.generative:
        with net.genGraph.as_default():
            initialize_uninitialized(net.genSess)
        net.genGraph.finalize()

    net.mainGraph.finalize()

    # Save w/ structure
    with net.mainGraph.as_default():
        saver.save(net.sess, "%s/%s.ckpt"%(net.checkpoints_dir, net.clf_name))

    # Generate validation set
    Xval,Yval_seg,Yval_det = feed.validation_set(nPatchesInValSet)

    it = 0
    best_val = 1e20
    # Start main training loop
    if( threaded ):
        while(feed.isOver == False):
            batch_X,batch_Y_seg,batch_Y_det = feed.getBatch()
            trainingStep.run(session=net.sess, feed_dict={net.X: batch_X, net.target_seg: batch_Y_seg, net.target_det: batch_Y_det})
            if( it % 100 == 0 ):
                with net.mainGraph.as_default():
                    [summ, lv] = net.sess.run([merged, loss], feed_dict={net.X: Xval, net.target_seg: Yval_seg, net.target_det: Yval_det})
                    train_writer.add_summary(summ, it)
                    saver.save(net.sess, "%s/%s.ckpt"%(net.checkpoints_dir,net.clf_name))
                    if( lv < best_val ):
                        best_val = lv
                        saver.save(net.sess, "%s/%s_best.ckpt"%(net.checkpoints_dir,net.clf_name))  # Save current best on validation set
            it += 1
    else:
        for batch_X,batch_Y_seg,batch_Y_det in feed.next_batch(batch_size, epochs):
            trainingStep.run(session=net.sess, feed_dict={net.X: batch_X, net.target_seg: batch_Y_seg, net.target_det: batch_Y_det})
            if( it % 100 == 0 ):
                with net.mainGraph.as_default():
                    [summ, lv] = net.sess.run([merged, loss], feed_dict={net.X: Xval, net.target_seg: Yval_seg, net.target_det: Yval_det})
                    train_writer.add_summary(summ, it)
                    saver.save(net.sess, "%s/%s.ckpt"%(net.checkpoints_dir,net.clf_name))
                    if( lv < best_val ):
                        best_val = lv
                        saver.save(net.sess, "%s/%s_best.ckpt"%(net.checkpoints_dir,net.clf_name))  # Save current best on validation set
            it += 1

    # Save at the end
    with net.mainGraph.as_default():
        saver.save(net.sess, "%s/%s.ckpt"%(net.checkpoints_dir,net.clf_name))

'''
Restore a network & session from meta graph & checkpoint 
'''
def restore(network_path):
    tf.reset_default_graph()
    sess = tf.Session()
    saver = tf.train.import_meta_graph('%s.meta'%network_path)
    saver.restore(sess, network_path)

    return sess, saver