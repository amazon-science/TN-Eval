# TN-Eval

This repository contains the code for our ACL 2025 paper: [TN-Eval: Rubric and Evaluation Protocols for Measuring the Quality of Behavioral Therapy
Notes](https://arxiv.org/abs/2503.20648).

**Authors**: 
[Raj Sanjay Shah](https://raj-sanjay-shah.github.io/), 
[Lei Xu](leixx.io), 
[Qianchu Liu](https://qianchu.github.io/), 
[Jon Burnsky](https://jburnsky.github.io/linguist/), 
Drew Bertagnolli,
[Chaitanya Shivade](https://cshivade.github.io/)

## Introduction
TN-Eval provides tools for generating behavioral therapy notes using large language models (LLMs) and evaluating them via automatic, rubric-based protocols.

## Quick Start

**Download Data**

Download [AnnoMI](https://github.com/uccollab/AnnoMI) data from https://github.com/uccollab/AnnoMI/raw/refs/heads/main/AnnoMI-full.csv and save it as `data/AnnoMI-full.csv`. 

**Generate Notes**

```bash
python3 src/generate_soap_note.py --input data/AnnoMi-full.csv --output data/llm_notes/
```

**Run Automatic Evaluations**
```base
python3 src/run_metrics_reference_free.py \
    --note data/llm_notes/outputs_annomi_llama31_70B_high.json \
    --output data/llm_notes/utputs_annomi_llama31_70B_high_with_eval.json
```

## Human Notes and Evaluations
You can find all data artifacts in our companion repository: [TN-Eval-Data](https://github.com/amazon-science/TN-Eval-Data).

This includes:
- Human-written therapy notes
- Human evaluations of human notes and LLM-generated notes
- Automatic evaluations using LLaMA and Mistral models


## Citation

If you use our data, please cite

```
@inproceedings{shah2025tneval,
  title={TN-Eval: Rubric and Evaluation Protocols for Measuring the Quality of Behavioral Therapy Notes},
  author={Shah, Raj Sanjay and Xu, Lei and Liu, Qianchu and Burnsky, Jon and Bertagnolli, Drew and Shivade, Chaitanya},
  booktitle={Proceedings of the 63nd Annual Meeting of the Association for Computational Linguistics: Industry Track},
  year={2025}
}
```

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.

