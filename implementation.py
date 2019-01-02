import tensorflow as tf

BATCH_SIZE = 128
MAX_WORDS_IN_REVIEW = 100  # Maximum length of a review to consider
EMBEDDING_SIZE = 50  # Dimensions for each word vector

stop_words = set({'ourselves', 'hers', 'between', 'yourself', 'again',
                  'there', 'about', 'once', 'during', 'out', 'very', 'having',
                  'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its',
                  'yours', 'such', 'into', 'of', 'most', 'itself', 'other',
                  'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him',
                  'each', 'the', 'themselves', 'below', 'are', 'we',
                  'these', 'your', 'his', 'through', 'don', 'me', 'were',
                  'her', 'more', 'himself', 'this', 'down', 'should', 'our',
                  'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had',
                  'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them',
                  'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does',
                  'yourselves', 'then', 'that', 'because', 'what', 'over',
                  'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you',
                  'herself', 'has', 'just', 'where', 'too', 'only', 'myself',
                  'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being',
                  'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it',
                  'how', 'further', 'was', 'here', 'than'})

def preprocess(review):
    """
    Apply preprocessing to a single review. You can do anything here that is manipulation
    at a string level, e.g.
        - removing stop words
        - stripping/adding punctuation
        - changing case
        - word find/replace
    RETURN: the preprocessed review in string form.
    """
    
    #Stripping punctuations

    no_punc = "".join(c for c in review if c not in ('!','.',':','?','\'','"'))
    
    #Lowercaseing

    lower_case = no_punc.lower()

    #Removing Stop Words
    
    words = lower_case.split()
    processed_review = [w for w in words if w not in stop_words]

    return processed_review



def define_graph():
    """
    Implement your model here. You will need to define placeholders, for the input and labels,
    Note that the input is not strings of words, but the strings after the embedding lookup
    has been applied (i.e. arrays of floats).

    In all cases this code will be called by an unaltered runner.py. You should read this
    file and ensure your code here is compatible.

    Consult the assignment specification for details of which parts of the TF API are
    permitted for use in this function.

    You must return, in the following order, the placeholders/tensors for;
    RETURNS: input, labels, optimizer, accuracy and loss
    """

    input_data = tf.placeholder(tf.float32,shape=[BATCH_SIZE,MAX_WORDS_IN_REVIEW,EMBEDDING_SIZE],name="input_data")
    labels = tf.placeholder(tf.float32,shape=[BATCH_SIZE,2],name="labels")
    
    dropout_keep_prob = tf.placeholder_with_default(tf.convert_to_tensor(1.0, dtype=tf.float32),shape=[],name="keep_prob")

    GRUCell = tf.nn.rnn_cell.GRUCell(128)
    GRUCell = tf.nn.rnn_cell.DropoutWrapper(cell=GRUCell,output_keep_prob=dropout_keep_prob)
    outputs, state = tf.nn.dynamic_rnn(cell=GRUCell,inputs=input_data,dtype=tf.float32)
    fullyConnected1 = tf.contrib.layers.fully_connected(outputs[:,-1],128,weights_initializer=tf.truncated_normal_initializer(),biases_initializer=tf.zeros_initializer(),activation_fn=tf.nn.softmax)
    fullyConnected1 = tf.layers.dropout(fullyConnected1,rate=1-dropout_keep_prob)
    logits = tf.contrib.layers.fully_connected(fullyConnected1,2,weights_initializer=tf.truncated_normal_initializer(),biases_initializer=tf.zeros_initializer(),activation_fn=None)
    prediction = tf.nn.softmax(logits)
    correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
    accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32),name="accuracy")
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels),name="loss")
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    return input_data, labels, dropout_keep_prob, optimizer, accuracy, loss









