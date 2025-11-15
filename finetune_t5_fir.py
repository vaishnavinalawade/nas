"""
finetune_t5_fir.py
Fine-tune a T5 model for FIR + Psych report generation.
"""
import argparse
from datasets import load_dataset
from transformers import T5TokenizerFast, T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments

def tokenise(batch, tok):
    inputs = tok(batch['input'], truncation=True, padding='max_length', max_length=512)
    labels = tok(batch['target'], truncation=True, padding='max_length', max_length=256)
    inputs['labels'] = labels['input_ids']
    return inputs

def main(args):
    data_files = {'train': args.train}
    if args.val: data_files['validation'] = args.val
    ds = load_dataset('json', data_files=data_files)
    tok = T5TokenizerFast.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    dsm = ds.map(lambda b: tokenise(b, tok), batched=True, remove_columns=ds['train'].column_names)
    ta = Seq2SeqTrainingArguments(
        output_dir=args.outdir, per_device_train_batch_size=8, per_device_eval_batch_size=8,
        predict_with_generate=True, logging_steps=100, eval_steps=500, save_total_limit=3,
        num_train_epochs=3, fp16=False, save_strategy="epoch", evaluation_strategy="epoch")
    trainer = Seq2SeqTrainer(model=model, args=ta, train_dataset=dsm['train'],
                             eval_dataset=dsm.get('validation'), tokenizer=tok)
    trainer.train()
    trainer.save_model(args.outdir); tok.save_pretrained(args.outdir)
    print("Saved to", args.outdir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', required=True)
    ap.add_argument('--val', required=False)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--model_name', default='t5-small')
    args = ap.parse_args()
    main(args)
