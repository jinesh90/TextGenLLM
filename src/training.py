import torch
import tiktoken
from torch.utils.data import DataLoader
from components import GPTModel,GPTDatasetV1
from config import GPT_2_mini_CONFIG_124M


# helper methods
def text_to_token(text, tokenizer,device):
  encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
  encoded_tensor = torch.tensor(encoded).unsqueeze(0).to(device) # add batch dimension
  return encoded_tensor

def token_to_text(token, tokenizer):
  decoded_text = tokenizer.decode(token.squeeze(0).tolist()) # removes batch dimesion
  return decoded_text


# define some helper functions for training
def generate_text_simple(model, idx,
                         max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

def create_dataloader_v1(txt, batch_size=2, max_length=1024,
        stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt,tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=0
    )

    return dataloader

def calc_loss_batch(input_batch, target_batch, tmodel, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = tmodel(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss

def calc_loss_loader(data_loader, tmodel, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
      input_batch = input_batch.to(device)
      target_batch = target_batch.to(device)
      if i < num_batches:
            loss = calc_loss_batch(
                input_batch, target_batch, tmodel, device
            )
            total_loss += loss.item()
      else:
        break
    return total_loss / num_batches

# define model eval function
def evaluate_model(tmodel, train_loader, val_loader, device, eval_iter):
    tmodel.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, tmodel, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(
            val_loader, tmodel, device, num_batches=eval_iter
        )
    tmodel.train()
    return train_loss, val_loss

def generate_and_print_sample(tmodel, tokenizer, device, start_context):
    tmodel.eval()
    context_size = tmodel.pos_emb.weight.shape[0]
    encoded = text_to_token(start_context, tokenizer, device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=tmodel, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    tmodel.train()

# training function
def train_model_simple(tmodel, train_loader, val_loader,
                       optimizer, device, num_epochs, eval_freq,
                       eval_iter, start_context, tokenizer):
  # for tracking losses and seen tokens
  train_loss_track = []
  val_loss_track = []
  token_seen = []

  # for stpes and total token seen
  token_display, global_step = 0,1

  for epoch in range(num_epochs):
    # set model on train mode
    tmodel.train()

    tmodel.to(device)

    for input_batch, target_batch in train_loader:

      input_batch, target_batch = input_batch.to(device), target_batch.to(device)


      # set optimizer gradient to zero
      optimizer.zero_grad()

      # calculate the loss, see function define above

      loss = calc_loss_batch(input_batch, target_batch, tmodel, device)

      # adjust the weight backprops
      loss.backward()

      # change optimizer step
      optimizer.step()

      token_display += input_batch.numel()
      global_step += 1


      # now only evalute model by set freq to save time
      if global_step % eval_freq == 0:
        train_loss, val_loss = evaluate_model(
                    tmodel, train_loader, val_loader, device, eval_iter)
        train_loss_track.append(train_loss)
        val_loss_track.append(val_loss)
        token_seen.append(token_display)
        print(input_batch.device,target_batch.device)
        print(f"Ep {epoch+1} (Step {global_step:06d}): "
              f"Train loss {train_loss:.3f}, "
              f"Val loss {val_loss:.3f}"
        )


    generate_and_print_sample(
        tmodel, tokenizer, device, start_context
    )

  return train_loss_track, val_loss_track, token_seen



# training starts from here
def main():

    # set seed
    torch.manual_seed(123)

    config = GPT_2_mini_CONFIG_124M
    with open(config['data_file'], "r", encoding="utf-8") as file:
        text_data = file.read()
        tokenizer = tiktoken.get_encoding("gpt2")
        # now split data between train and validation with ratio 90,10 without any shuffling or any modification
        train_ratio = 0.9
        split_idx = int(train_ratio * len(text_data))
        train_data = text_data[:split_idx]
        val_data = text_data[split_idx:]


        # create data loader with train,val data
        train_loader = create_dataloader_v1(
            train_data,
            batch_size=2,
            max_length=config["context_length"],
            stride=config["context_length"],
            drop_last=True,
            shuffle=True,
            num_workers=0
        )
        val_loader = create_dataloader_v1(
            val_data,
            batch_size=2,
            max_length=config["context_length"],
            stride=config["context_length"],
            drop_last=False,
            shuffle=False,
            num_workers=0
        )

        # define model
        gpt_model = GPTModel(config)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        gpt_model.to(device)

        optimizer = torch.optim.AdamW(
            gpt_model.parameters(),
            lr=0.0004, weight_decay=0.1
        )
        num_epochs = 10
        train_losses, val_losses, tokens_seen = train_model_simple(
            gpt_model, train_loader, val_loader, optimizer, device,
            num_epochs=num_epochs, eval_freq=5, eval_iter=5,
            start_context="Light where you can", tokenizer=tokenizer
        )

# start training
main()


