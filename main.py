from flask import Flask, render_template, request, url_for
import time
import mxnet as mx
import gluonnlp as nlp
nlp.utils.check_version('0.8.0')


ctx = mx.cpu()  # mx.gpu(0)
bsize = 10
maxl = 500
temp = 0.85
num_print = 1


class LMDecoder(object):
    def __init__(self, model):
        self._model = model

    def __call__(self, inputs, states):
        outputs, states = self._model(mx.nd.expand_dims(inputs, axis=0), states)
        return outputs[0], states

    def state_info(self, *arg, **kwargs):
        return self._model.state_info(*arg, **kwargs)


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index5.html')


@app.route('/predict', methods=["POST"])
def predict():
    start = time.time()
    raw_text = request.form['rawtext']
    bos = raw_text.split()
    lm_model, vocab = nlp.model.get_model(name='awd_lstm_lm_1150',   # standard_lstm_lm_1500, awd_lstm_lm_1150, big_rnn_lm_2048_512, transformer_en_de_512, bert_24_1024_16, elmo_2x4096_512_2048cnn_2xhighway
                                          dataset_name='wikitext-2',  # gbw, wikitext-2
                                          pretrained=True,
                                          ctx=ctx)

    decoder = LMDecoder(lm_model)

    eos_id = vocab['.']
    bos_ids = [vocab[ele] for ele in bos]
    begin_states = lm_model.begin_state(batch_size=1, ctx=ctx)
    if len(bos_ids) > 1:
        _, begin_states = lm_model(mx.nd.expand_dims(mx.nd.array(bos_ids[:-1]), axis=1),
                                   begin_states)
    inputs = mx.nd.full(shape=(1,), ctx=ctx, val=bos_ids[-1])

    def generate_sequences(sampler, inputs, begin_states, num_print_outcomes):
        samples, scores, valid_lengths = sampler(inputs, begin_states)
        samples = samples[0].asnumpy()
        scores = scores[0].asnumpy()
        valid_lengths = valid_lengths[0].asnumpy()

        for i in range(num_print_outcomes):
            sentence = bos[:-1]

            for ele in samples[i][:valid_lengths[i]]:
                sentence.append(vocab.idx_to_token[ele])

            ans = ' '.join(sentence)
            score = '{:.2f}'.format(scores[i])
            return ans, score

    seq_sampler = nlp.model.SequenceSampler(beam_size=bsize,
                                            decoder=decoder,
                                            eos_id=eos_id,
                                            max_length=maxl,
                                            temperature=temp)

    my_prediction, pred_score = generate_sequences(seq_sampler, inputs, begin_states, num_print)
    end = time.time()
    final_time = '{:.2f}'.format((end-start))
    return render_template('index5.html', prediction=my_prediction, pred_score=pred_score, final_time=final_time)


if __name__ == '__main__':
    app.run(debug=True)
