import torch as t
import transformer_lens
from matplotlib import pyplot as plt
import string
import random


def default_model():
    return transformer_lens.HookedTransformer.from_pretrained(
        "gpt2-small", 
        device=t.device('mps')
    )


# We're going to generate a sequence of tokens of the form
#    '\n', {token}, ' ->', {category}, '\n', {token}, ' ->', ...
# Where {category} is a random choice of ' yes' or ' no' and
# {token} is drawn uniformly from either `good_tkns` or `bad_tkns` as a function
# of the following {category}.
#
# Predicting this sequence requires being aware of the two categories for {token}.
# Can we find a "feature" being computed somewhere that describes the category
# of a {token}?
class CategoryTask:
    def __init__(self, model):
        self.model = model
        self.good_tkns = [model.to_single_token(c) for c in string.ascii_lowercase]
        self.bad_tkns = [model.to_single_token(c) for c in string.ascii_uppercase]
        self.yes_tkn = model.to_single_token(' yes')
        self.no_tkn = model.to_single_token(' no')
        self.newline_tkn = model.to_single_token('\n')
        self.arrow_tkn = model.to_single_token(' ->')

        self.special_tokens = [self.newline_tkn, self.arrow_tkn, self.yes_tkn, self.no_tkn]
        self.special_token_names = [
            '\\n', '->', 'yes', 'no'
        ]

    def make_input_tokens(self, n=10):
        random.seed(4242)
        categories = [(self.yes_tkn, self.good_tkns), (self.no_tkn, self.bad_tkns)]
        sequence = []
        for _ in range(n):
            cat = random.choice(categories)
            tkn = random.choice(cat[1])
            sequence.extend([self.newline_tkn, tkn, self.arrow_tkn, cat[0]])
        return t.tensor(sequence)

    @staticmethod
    def get_prediction_indices(n=10):
        return 4 * t.arange(0, n, dtype=t.int) + 3

    def show_logits(self, logits):
        # logits has shape (seq, vocab)
        # truth has shape (seq) with values indexing vocab
        special_logits = logits[:, self.special_tokens]
        plt.matshow(t.softmax(special_logits, dim=-1).t()[:, 1:])
        plt.yticks(range(4), labels=self.special_token_names)

    def prediction_success(self, input_tokens, logits):
        prediction_indices = CategoryTask.get_prediction_indices(input_tokens.shape[0] // 4)
        prediction = t.argmax(logits[prediction_indices - 1, :][:, [self.yes_tkn, self.no_tkn]], dim=-1)
        truth = (input_tokens[prediction_indices] == self.no_tkn).to(t.int)
        return prediction == truth.cpu()


def get_activation_grads(model, input_tokens, layers, loss_indices=None):
    if loss_indices is None:
        loss_indices = t.arange(input_tokens.shape[0])

    result = {}
    def grad_hook(value, hook:transformer_lens.hook_points.HookPoint):
        result[hook.layer()] = value[0, ...]

    bwd_hooks = [
        (f'blocks.{i}.hook_resid_post', grad_hook)
        for i in layers
    ]

    with model.hooks([], bwd_hooks) as hooked_model:
        print("forward pass...")
        logits = hooked_model.forward(input_tokens)
        print("loss...")
        loss = model.loss_fn(logits[:, loss_indices, :], input_tokens[loss_indices].reshape((1, -1)))
        print("backward pass...")
        loss.backward()

    return result


# cv.tokens.colored_tokens(
#     model.to_str_tokens(input_tokens), 
#     logits[:, yes_tkn] - logits[:, no_tkn]
# )


# %%
# Show accuracy at predicting the yes/no tokens.
# Seems like gpt-2 figures out the pattern and can distinguish letters from numbers.

# plt.matshow(prediction_success(input_tokens, logits).reshape((1, -1)))
# plt.yticks([])
# plt.show()


# %%
# Okay, so far we've set up our really simple task and checked that GPT-2-small
# isn't terrible at it. Now let's try out our strange proposal of how we might
# infer the presence of feature vectors.

# We need to identify some loss function that the parameters are approximately
# stationary with respect to. We'll consider cross-entropy loss between logits 
# and tokens late in the sequence. Then, we need to access gradients with respect
# to this loss.

# %%
# input_tokens = category_task(150)
# layers = t.arange(0, 12)
# grads = get_activation_grads(
#     input_tokens, layers,
#     loss_indices=t.arange(50 * 4, 150 * 4)
# )

# # %%
# grads_tensor = t.stack([grads[i] for i in layers.tolist()]).to('cpu')


# # %%
# t.save(grads_tensor, "grads_1")
