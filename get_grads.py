# %%
import torch as t
import transformer_lens
import circuitsvis as cv
from matplotlib import pyplot as plt
import noether


# %%
device = t.device("mps")

model = transformer_lens.HookedTransformer.from_pretrained(
    "gpt2-small", 
    device=device,
)

# %%
task = noether.CategoryTask(model)


# %%
# Compute logits
input_tokens = task.make_input_tokens(150)
logits = model(input_tokens)[0, :, :].data.cpu() # (seq, vocab)


# %%
# View logits.
task.show_logits(logits[:, :])


# %%
# Show accuracy at predicting the yes/no tokens.
# Seems like gpt-2 figures out the pattern and can distinguish letters from numbers.
plt.matshow(task.prediction_success(input_tokens, logits)[None, :])
plt.yticks([], [])
plt.show()


# %%
input_tokens = category_task(150)
layers = t.arange(0, 12)
grads = get_activation_grads(
    input_tokens, layers,
    loss_indices=t.arange(50 * 4, 150 * 4)
)

# %%
grads_tensor = t.stack([grads[i] for i in layers]).to('cpu')


# %%
grads_tensor
