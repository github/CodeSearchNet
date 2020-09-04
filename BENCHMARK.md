> ## The Challenge has been concluded
> No new submissions to the benchmark will be accepted. However, we would like
> to encourage practitioners and researchers to continue using
> the dataset and the human relevance annotations. Please see the
> [main README](/README.md) for more information.

## Submitting runs to the benchmark

The [Weights & Biases (W&B)](https://www.wandb.com) [benchmark](https://app.wandb.ai/github/CodeSearchNet/benchmark) tracks and compares models trained on the CodeSearchNet dataset by the global machine learning research community. Anyone is welcome to submit their results for review.

The leaderboard is available at <https://app.wandb.ai/github/codesearchnet/benchmark/leaderboard>.

## Submission process

### Requirements

There are a few requirements for submitting a model to the benchmark.
- You must a have a run logged to [W&B](https://app.wandb.ai).
- Your run must have attached inference results in a file named  `model_predictions.csv`. You can view all the files attached to a given run in the browser by clicking the "Files" icon from that run's main page. 
- The schema outlined in the submission format section below must be strictly followed. 

### Submission format

*To submit from our baseline model, skip to the [training the baseline model](#training-the-baseline-model-optional) section below.*

A valid submission to the CodeSeachNet Challenge requires a file named **model_predictions.csv** with the following fields: `query`, `language`, `identifier`, and `url`:

* `query`: the textual representation of the query, e.g. "int to string" .  
* `language`: the programming language for the given query, e.g. "python".  This information is available as a field in the data to be scored.
* `identifier`: this is an optional field that can help you track your data
* `url`: the unique GitHub URL to the returned results, e.g. "https://github.com/JamesClonk/vultr/blob/fed59ad207c9bda0a5dfe4d18de53ccbb3d80c91/cmd/commands.go#L12-L190" . This information is available as a field in the data to be scored.
      
For further background and instructions on the submission process, see [the root README](README.md).

The row order corresponds to the result ranking in the search task. For example, if in row 5 there is an entry for the Python query "read properties file", and in row 60 another result for the Python query "read properties file", then the URL in row 5 is considered to be ranked higher than the URL in row 60 for that query and language.

Here is an example: 

| query                 | language | identifier                        | url                                                                                                                                                   |
| --------------------- | -------- | --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| convert int to string | python   | int_to_decimal_str                | https://github.com/raphaelm/python-sepaxml/blob/187b699b1673c862002b2bae7e1bd62fe8623aec/sepaxml/utils.py#L64-L76                                     |
| convert int to string | python   | str_to_int_array                  | https://github.com/UCSBarchlab/PyRTL/blob/0988e5c9c10ededd5e1f58d5306603f9edf4b3e2/pyrtl/rtllib/libutils.py#L23-L33                                   |
| convert int to string | python   | Bcp47LanguageParser.IntStr26ToInt | https://github.com/google/transitfeed/blob/eb2991a3747ba541b2cb66502b305b6304a1f85f/extensions/googletransit/pybcp47/bcp47languageparser.py#L138-L139 |
| convert int to string | python   | PrimaryEqualProof.to_str_dict     | https://github.com/hyperledger-archives/indy-anoncreds/blob/9d9cda3d505c312257d99a13d74d8f05dac3091a/anoncreds/protocol/types.py#L604-L613            |
| convert int to string | python   | to_int                            | https://github.com/mfussenegger/cr8/blob/a37d6049f1f9fee2d0556efae2b7b7f8761bffe8/cr8/cli.py#L8-L23                                                   |
| how to read .csv file in an efficient way? | ruby | Icosmith.Font.generate_scss                | https://github.com/tulios/icosmith-rails/blob/e73c11eaa593fcb6f9ba93d34fbdbfe131693af4/lib/icosmith-rails/font.rb#L80-L88             |
| how to read .csv file in an efficient way? | ruby | WebSocket.Extensions.valid_frame_rsv       | https://github.com/faye/websocket-extensions-ruby/blob/1a441fac807e08597ec4b315d4022aea716f3efc/lib/websocket/extensions.rb#L120-L134 |
| how to read .csv file in an efficient way? | ruby | APNS.Pem.read_file_at_path                 | https://github.com/jrbeck/mercurius/blob/1580a4af841a6f30ac62f87739fdff87e9608682/lib/mercurius/apns/pem.rb#L12-L18                   |



### Submitting model predictions to W&B 

You can submit your results to the benchmark as follows:

1. Run a training job with any script (your own or the baseline example provided, with or without W&B logging).
2. Generate your own file of model predictions following the format above and name it \`model_predictions.csv\`.
3. Upload a run to wandb with this \`model_predictions.csv\` file attached.

Our example script [src/predict.py](src/predict.py) takes care of steps 2 and 3 for a model training run that has already been logged to W&B, given the corresponding W&B run id, which you can find on the /overview page in the browser or by clicking the 'info' icon on a given run.

### Publishing your submission

You've now generated all the content required to submit a run to the CodeSearchNet benchmark. Using the W&B GitHub integration you can now submit your model for review via the web app.

You can submit your runs by visiting the run page and clicking on the overview tab:
![](https://github.com/wandb/core/blob/master/frontends/app/src/assets/run-page-benchmark.png?raw=true)

or by visiting the project page and selecting a run from the runs table:
![](https://app.wandb.ai/static/media/submit_benchmark_run.e286da0d.png)

### Result evaluation

Once you upload your \`model_predictions.csv\` file, W&B will compute the normalized discounted cumulative gain (NDCG) of your model's predictions against the human-annotated relevance scores.  Further details on the evaluation process and metrics are in [the root README](README.md). For transparency, we include the script used to evaluate submissions: [src/relevanceeval.py](src/relevanceeval.py)


### Training the baseline model (optional)

Replicating our results for the CodeSearchNet baseline is optional, as we encourage the community to create their own models and methods for ranking search results.  To replicate our baseline submission, you can start with the "Quickstart" instructions in the [CodeSearchNet GitHub repository](https://github.com/github/CodeSearchNet).  This baseline model uses [src/predict.py](src/predict.py) to generate the submission file.

Your run will be logged to W&B, within a project that will be automatically linked to this benchmark.

### Rules

**Only 1 submission to the benchmark leaderboard is allowed every 2 weeks.**  Our intention is not for participants to make many submissions to the leaderboard with different parameters -- as this kind of overfitting is counterproductive. There are no cash prizes and the idea is to learn from this dataset, for example, to apply the learned representations or utilize new techniques.
