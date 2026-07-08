```
python scripts/generate_pairs_from_neo4j.py --uri bolt://localhost:7687 --user neo4j --password rp_strong_pass_2025 --output ./data/raw/generated_pairs.parquet
```

```
python scripts/validate_pairs.py --input ./data/raw/generated_pairs.parquet --output data/validated/validated_pairs.parquet --uri neo4j://localhost:7687 --user neo4j --password rp_strong_pass_2025
```

```
 python scripts/diagnose.py --input data/raw/generated_pairs.parquet --uri bolt://localhost:7687 --user neo4j --password rp_strong_pass_2025
```

```
  python scripts/generate_pairs_from_neo4j.py --uri bolt://localhost:7687 --user neo4j --password rp_strong_pass_2025 --sample-patients 5000 --num-pairs 5000 --output ./data/raw/generated_pairs.parquet
```

```
  python scripts/train_lora.py --train-data ./data/splits/train.parquet --val-data   ./data/splits/val.parquet --num-epochs 3 --batch-size 2
```

```
  python scripts/eval_runner.py --model ./lora_adapter_v1 --eval  data/eval/eval.json --uri   bolt://localhost:7687 --user  neo4j --password rp_strong_pass_2025
```

```
python scripts/eval_runner.py --model ./lora_adapter_v1 --eval  data/eval/eval.json --uri   bolt://localhost:7687 --user  neo4j --password rp_strong_pass_2025
```

```
python scripts/validate_pairs.py --input  data/raw/generated_pairs.parquet --output data/validated/validated_pairs.parquet --uri bolt://localhost:7687 --user neo4j --password rp_strong_pass_2025    --build-splits
```

```
python scripts/eval_runner.py --model ./lora_adapter_v2 --eval  data/eval/eval.json --uri   bolt://localhost:7687 --user  neo4j --password rp_strong_pass_2025
```

```
python demo_inference.py --model ./lora_adapter_v2 --uri bolt://localhost:7687 --user neo4j --password rp_strong_pass_2025
```

```
python scripts/demo_inference.py --uri bolt://localhost:7687 --user neo4j --password rp_strong_pass_2025 --demo
```


### Smoke Test
```
python scripts/train_lora.py --train-data data/splits/train.parquet --val-data   data/splits/val.parquet --output-dir ./lora_smoke_test --num-epochs 1 --smoke-test --skip-gguf
```

python scripts/train_lora.py --train-data data/splits/train.parquet --val-data   data/splits/val.parquet  --output-dir ./lora_smoke_test --num-epochs 1 --smoke-test --skip-gguf

### If smoke test shows loss < 1.5 and no format errors, run the full training
```
python scripts/train_lora.py \
    --train-data data/splits/train.parquet \
    --val-data   data/splits/val.parquet \
    --output-dir ./lora_adapter_v3 \
    --num-epochs 3
```


# After training — quick sanity check before full eval
```
python scripts/quick_check.py --model ./lora_adapter_v3
```

# Full eval
```
python scripts/eval_runner.py \
    --model ./lora_adapter_v3 \
    --eval  data/eval/eval.json \
    --uri   bolt://localhost:7687 --user neo4j --password <pw>
```


## Demo Olama
python scripts/demo_ollama.py --uri bolt://localhost:7687 --user neo4j --password rp_strong_pass_2025 --demo

python demo_ollama.py \
    --model text2cypher \
    --uri bolt://localhost:7687 \
    --user neo4j --password rp_strong_pass_2025 \
    --demo

```
python scripts/demo_inference.py \
    --model ./lora_adapter_v3 \
    --uri bolt://localhost:7687 \
    --user neo4j --password rp_strong_pass_2025
```



python export_to_ollama.py --adapter ./lora_adapter_v3 --tag rp-cypher-v3


