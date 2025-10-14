# System related imports.
import os
import string

# Pytorch related imports.
import torch
from torch.nn import functional as F
from transformers import BertTokenizer
from transformers import BertForMaskedLM
from transformers import XLNetTokenizer
from transformers import XLNetModel
from transformers import AutoModelWithLMHead
from transformers import AutoTokenizer
import logging


def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> torch.Tensor:
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


# Declare global variables.
no_words_to_be_predicted = globals()
select_model = globals()
enter_input_text = globals()


def set_model_config(**kwargs):
    for key, value in kwargs.items():
        print("{0} = {1}".format(key, value))

    no_words_to_be_predicted = list(kwargs.values())[0] # integer values
    select_model = list(kwargs.values())[1] # possible values = 'bert' or 'gpt' or 'xlnet'
    enter_input_text = list(kwargs.values())[2] # only string

    return no_words_to_be_predicted, select_model, enter_input_text


def load_model(model_name):
    try:
        if model_name.lower() == "bert":
            bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased').eval()
            return bert_tokenizer, bert_model
        elif model_name.lower() == "gpt":
            gpt_tokenizer = AutoTokenizer.from_pretrained("gpt2")
            gpt_model = AutoModelWithLMHead.from_pretrained("gpt2")
            return gpt_tokenizer, gpt_model
        else:
            xlnet_tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
            xlnet_model = XLNetModel.from_pretrained("xlnet-based-cased")
            return xlnet_tokenizer, xlnet_model
    except Exception as exp:
        print(exp)


# bert encode
def encode_bert(tokenizer, text_sentence, add_special_tokens=True):
    text_sentence = text_sentence.replace('<mask>', tokenizer.mask_token)
    # if <mask> is the last token, append a "." so that models dont predict punctuation.
    if tokenizer.mask_token == text_sentence.split()[-1]:
        text_sentence += ' .'
        input_ids = torch.tensor([tokenizer.encode(text_sentence, add_special_tokens=add_special_tokens)])
        mask_idx = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0]
    return input_ids, mask_idx


# bert decode
def decode_bert(tokenizer, pred_idx, top_clean):
    ignore_tokens = string.punctuation + '[PAD]'
    tokens = []
    for w in pred_idx:
        token = ''.join(tokenizer.decode(w).split())
        if token not in ignore_tokens:
            tokens.append(token.replace('##', ''))
    return '\n'.join(tokens[:top_clean])


# gpt encode
def encode_gpt(tokenizer, text_sentence, add_special_tokens=False):
    input_ids = tokenizer.encode(text_sentence, return_tensors="pt")
    return input_ids


# gpt decode
def decode_gpt(tokenizer, input_ids, pred, top_clean):
    filtered_next_token_logits = top_k_top_p_filtering(pred, top_k=top_clean, top_p=1.0)

    # sample
    probs = F.softmax(filtered_next_token_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=top_clean)
    generated = torch.cat([input_ids, next_token], dim=-1)  
    resulting_string = tokenizer.decode(generated.tolist()[0])
    return resulting_string


def get_all_predictions(text_sentence,  model_name, tokenizer, model, nr_words, top_clean=5):
    if model_name.lower() == "bert":
        input_ids, mask_idx = encode_bert(tokenizer, text_sentence)
        with torch.no_grad():
            predict = model(input_ids)[0]
            bert = decode_bert(tokenizer, predict[0, mask_idx, :].topk(nr_words).indices.tolist(), top_clean)
        return {'bert': bert}
    elif model_name.lower() == "gpt":
        input_ids = encode_gpt(tokenizer, text_sentence)
        with torch.no_grad():
            predict = model(input_ids)[0][:, -1, :]
        gpt = decode_gpt(tokenizer, input_ids, predict, top_clean)
        return {'gpt': gpt}


def get_prediction_end_of_sentence(input_text, model_name, model, tokenizer, nr_words):
    if model_name.lower() == "bert":
        input_text += ' <mask>'
        print(input_text)
        res = get_all_predictions(input_text, model_name, tokenizer, model, nr_words, top_clean=nr_words) 
        return res
    elif model_name.lower() == "gpt":
        print(input_text)
        res = get_all_predictions(input_text, model_name, tokenizer, model, nr_words, top_clean=nr_words)
        return res


if __name__ == "__main__":
    nr_words, model, text = set_model_config(no_words_to_be_predicted=2,
                                             select_model="bert",
                                             text="Why whisper what you")
    tokenizer, model = load_model(model)
    res = get_prediction_end_of_sentence(text, "bert", model, tokenizer, nr_words)["bert"]
    print("------bert results-------")
    print("result is: Why whisper what you {}".format(res.replace("\n", " ")))
    print("Correct is: Why whisper what you can shout")

    nr_words, model, text = set_model_config(no_words_to_be_predicted=2,
                                             select_model="gpt",
                                             text="Why whisper what you")
    tokenizer, model = load_model(model)
    res = get_prediction_end_of_sentence(text, "gpt", model, tokenizer, nr_words)["gpt"]
    print("-------GPT results-------")
    print("result is: {}".format(res.replace("\n", " ")))
    print("Correct is: Why whisper what you can shout")
