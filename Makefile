.DEFAULT_GOAL := help
PY ?= python3
SRC := src tests tools modal_app

# Every build target below is one package's CLI plus a subcommand, so the invocation is
# written once here. A family of subcommands is one target with a variable, never one target
# per subcommand: `make bench TASK=pos`, not `make bench-pos`. Where the members of a family
# are whole commands rather than subcommands, the documented target dispatches to them as
# prerequisites and they carry no `##`, so `make help` lists the family and not its members.
cli = $(PY) -m agbalu.$(1).cli

# Deploy only the module that owns the function about to be spawned. `modal_app.deploy`
# imports all five images, and on an account with no cached layers that is five torch
# installs before anything starts. Status, logs and cancel address a call by its id and the
# app by name, so none of them needs the other modules to be deployed.
deploy = modal deploy -m modal_app.$(1)

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: ## Editable install with dev extras
	$(PY) -m pip install -e ".[dev]"

lint: ## ruff check + format check
	ruff check $(SRC)
	ruff format --check $(SRC)

format: ## Apply ruff formatting and safe fixes
	ruff check --fix $(SRC)
	ruff format $(SRC)

typecheck: ## mypy, strict
	mypy

test: ## Full pytest suite. UNIT=1 skips integration, COV=1 adds the coverage report
	$(PY) -m pytest $(if $(UNIT),-m "not integration",) $(if $(COV),--cov --cov-report=term-missing,)

check: lint typecheck test ## The gate: lint + typecheck + test
	@echo "All checks passed."

registry: ## Validate the corpus registry. SIBLINGS=1 validates the Berber sibling registry
	$(call cli,registry) $(if $(SIBLINGS),--siblings resources/sibling_registry.yaml,resources/corpus_registry.yaml)

acquire: acquire-$(or $(TASK),fetch) ## Fetch sources. TASK=fetch|plan|verify|flores|siblings

acquire-fetch:
	$(PY) -m agbalu.acquire.cli fetch --tier core

acquire-plan:
	$(PY) -m agbalu.acquire.cli plan --tier core

acquire-verify:
	$(PY) -m agbalu.acquire.cli verify

acquire-flores:
	$(PY) -m agbalu.acquire.cli fetch --id hf.flores-plus-kab --force

acquire-siblings:
	$(PY) -m agbalu.acquire.cli fetch --siblings
	$(PY) -m agbalu.acquire.cli fetch --id hf.glotlid-model --id hf.nllb-lid218e

normalise: ## Normaliser. TASK=check (idempotence over the seed corpus)|evalset|text
	$(call cli,normalise) $(or $(TASK),check)

extract: ## Build the clean monolingual corpus from data/raw
	$(PY) -m agbalu.extract.cli build

clean: ## Delete all tool caches and bytecode
	@find . -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true
	@find . -name ".pytest_cache" -type d -prune -exec rm -rf {} + 2>/dev/null || true
	@find . -name ".mypy_cache" -type d -prune -exec rm -rf {} + 2>/dev/null || true
	@find . -name ".ruff_cache" -type d -prune -exec rm -rf {} + 2>/dev/null || true
	@find . -name ".hypothesis" -type d -prune -exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.pyc" -delete 2>/dev/null || true
	@rm -rf .coverage htmlcov coverage.xml build dist .benchmarks .tokenizer-cache
	@echo "Caches cleaned."

bench: ## TASK=audit|contamination|score|pos|lid|lid-sources|lid-audit|tifinagh|inflect. lid-audit takes MARKER=ṯ
	$(call cli,bench) $(or $(TASK),audit) $(if $(MARKER),--marker "$(MARKER)",)

bench-ceiling: ## What a perfect system scores against the uncorrected FLORES+ reference
	$(PY) tools/benchmark_ceiling.py

parallel: ## Build AƔBALU-Parallel v1. TASK=agreement partitions the mined NLLB pool
	$(call cli,parallel) $(or $(TASK),build)

lexicon: ## Build AƔBALU-Lexicon v1. TASK=build|validate (against gold UD)|coverage|analyse
	$(call cli,lexicon) $(or $(TASK),build) $(if $(filter coverage,$(TASK)),--unknown,)

pronunciations: ## Word-align the sentence-level G2P data into a pronunciation lexicon
	$(PY) -m agbalu.g2p.cli build

tokenizer: ## Train AƔBALU-Tok v1. STAGE=prepare|build|sweep|evaluate
	$(call cli,tokenizer) $(or $(STAGE),build)

model: ## Encoder. TASK=data|train (50 local steps, STEPS=n)|fill-mask|neighbours|analogy
	$(call cli,model) $(or $(TASK),data) $(if $(filter train,$(TASK)),--steps $(or $(STEPS),50),)

mt: ## NLLB corpus. TASK=corpus|pivot|consistency (pivot needs the models extra)
	$(call cli,mt) $(or $(TASK),corpus)

llm: ## Phase 11, GPU-free. TASK=fertility|holdout|mixture
	$(call cli,llm) $(or $(TASK),fertility) \
		$(if $(BLOCKS),--blocks $(BLOCKS),) $(if $(TOKENS),--tokens $(TOKENS),)

speech: ## Phase 5. TASK=corpus|vocabulary|lm (needs lmplz)|release|transcribe (AUDIO=path, needs the speech extra)
	$(call cli,speech) $(or $(TASK),corpus) $(if $(AUDIO),$(AUDIO),) \
		$(if $(LIMIT),--limit $(LIMIT),) $(if $(GREEDY),--greedy,) \
		$(if $(DEVICE),--device $(DEVICE),)

tts: ## Matoub front-end, GPU-free. TASK=validate|prompts|cycle (RESULT=<json> checks its control)
	$(call cli,tts) $(or $(TASK),validate) $(if $(RESULT),--result $(RESULT),)

punctuation: ## Restore marks and capitals on ASR output. TASK=corpus|train|evaluate|restore
	$(call cli,punctuation) $(or $(TASK),evaluate) \
		$(if $(TEXT),--text "$(TEXT)",) $(if $(SPLIT),--split $(SPLIT),) \
		$(if $(EPOCHS),--epochs $(EPOCHS),) $(if $(DEVICE),--device $(DEVICE),)

listen: ## Fadhma then punctuation, end to end on local audio. AUDIO=path LIMIT RUN GREEDY=1
	@$(PY) -m agbalu.speech.cli transcribe $(AUDIO) \
		$(if $(LIMIT),--limit $(LIMIT),) $(if $(GREEDY),--greedy,) \
	| $(PY) -m agbalu.punctuation.cli --run artifacts/runs/$(or $(RUN),punctuation-v2) restore

tifinagh: ## Script conversion with Juba-27M. TASK=convert TEXT="ⴰⵣⵓⵍ"|evaluate. SPLIT LIMIT
	$(call cli,tifinagh) $(or $(TASK),evaluate) \
		$(if $(TEXT),--text "$(TEXT)",) $(if $(SPLIT),--split $(SPLIT),) \
		$(if $(LIMIT),--limit $(LIMIT),)

RELEASE ?= artifacts/release
CHECKPOINTS ?= artifacts/checkpoints

# Datasets go through one path, because the path is where the checks are: the Hub validates
# card metadata on *render*, so a bad `task_categories` or `task_ids` publishes silently and
# is visible only to whoever opens the page. Models are staged per repo, since each has a
# different weight format.
DATASET_REPOS := bench lex sentiment inflect tifinagh
MODEL_REPOS := encoder tokenizer mt juba
REPOS := $(MODEL_REPOS) $(DATASET_REPOS) org

release: $(addprefix release-,$(or $(REPO),$(REPOS))) ## Stage repos under $(RELEASE). REPO=encoder|tokenizer|mt|juba|fadhma|belaid|bench|lex|sentiment|inflect|tifinagh|org
	@echo 'staged under $(RELEASE) — review, then push each with: hf upload <repo> <dir> .'

$(addprefix release-,$(DATASET_REPOS)): release-%:
	$(PY) -m tools.export_datasets --only $* --out $(RELEASE)

release-org:
	@mkdir -p $(RELEASE)/org-README
	@cp docs/cards/organization.md $(RELEASE)/org-README/README.md
	@echo '$(RELEASE)/org-README -> hf upload agbalu/README $(RELEASE)/org-README . --repo-type=space'

# `stage_hub` second, always: neither architecture is native to transformers, so without
# the standalone modelling code and the `auto_map` beside them the weights publish with
# nothing a downloader can construct.
release-encoder:
	$(PY) -m tools.export_release --run agbalu-encoder-v1 --preset kab \
		--out $(RELEASE)/Masinissa-31M --card docs/cards/masinissa-31m.md \
		--tokenizer artifacts/tokenizer/agbalu-tok-base-16k.model
	$(PY) -m tools.stage_hub --repo masinissa --dir $(RELEASE)/Masinissa-31M

release-mt:
	$(PY) -m tools.stage_mt_release --source $(CHECKPOINTS)/Amrouche-1.3B \
		--out $(RELEASE)/Amrouche-1.3B --card docs/cards/amrouche-1.3b.md

release-juba:
	$(PY) -m tools.export_checkpoint --source $(CHECKPOINTS)/Juba-27M/juba_final.pt \
		--out $(RELEASE)/Juba-27M --card docs/cards/juba-27m.md
	$(PY) -m tools.stage_hub --repo juba --dir $(RELEASE)/Juba-27M

# The ASR checkpoint is resumable state and carries no architecture, so the config, the
# feature extractor and the tokenizer are written first from the same constants the
# training loop builds the model with. Every file `speech release` writes is staged here,
# and tests/unit/test_speech_release.py asserts this recipe names all of them.
release-fadhma:
	$(call cli,speech) release
	$(PY) -m tools.export_checkpoint --source artifacts/asr/best.pt \
		--out $(RELEASE)/Fadhma-300M --card docs/cards/fadhma-300m.md \
		--config artifacts/asr/config.json \
		--extra artifacts/asr/vocab.json --extra artifacts/asr/5gram.klm \
		--extra artifacts/asr/preprocessor_config.json \
		--extra artifacts/asr/tokenizer_config.json \
		--extra artifacts/asr/added_tokens.json

# Same shape as Fadhma and for the same reason: the restorer's checkpoint is resumable state
# and carries no architecture, so the config is written first from the preset and the label
# tuples the heads were sized by. Out of the default `REPOS` because it needs a checkpoint
# pulled off the volume, exactly as Fadhma does.
PUNCTUATION_RUN ?= punctuation-v2
release-belaid:
	$(call cli,punctuation) release
	$(PY) -m tools.export_checkpoint --source artifacts/runs/$(PUNCTUATION_RUN)/best.pt \
		--out $(RELEASE)/Belaid-31M --card docs/cards/belaid-31m.md \
		--config artifacts/punctuation/config.json \
		--extra artifacts/tokenizer/agbalu-tok-base-16k.model \
		--derived position_indices
	$(PY) -m tools.stage_hub --repo belaid --dir $(RELEASE)/Belaid-31M

release-tokenizer:
	@mkdir -p $(RELEASE)/Mammeri-Tok
	@cp artifacts/tokenizer/agbalu-tok-*.model artifacts/tokenizer/agbalu-tok-*.vocab \
		artifacts/tokenizer/agbalu-tok-*.metadata.json $(RELEASE)/Mammeri-Tok/
	@cp data/processed/tokenizer/agbalu-tok-v1.eval.json $(RELEASE)/Mammeri-Tok/sweep.json
	@cp docs/cards/mammeri-tok.md $(RELEASE)/Mammeri-Tok/README.md
	@echo "$(RELEASE)/Mammeri-Tok: $$(ls $(RELEASE)/Mammeri-Tok | wc -l | tr -d ' ') files"

UPLOADS := corpus bench mt speech llm tts punctuation

modal-upload: $(addprefix modal-upload-,$(or $(TASK),$(UPLOADS))) ## Push what a container reads off the volumes. TASK=corpus|bench|mt|speech|llm|tts|punctuation|encoder. Tifinagh fetches its own

modal-upload-corpus:
	modal run -m modal_app.train::upload_corpus

modal-upload-bench:
	modal run -m modal_app.bench::upload_bench

modal-upload-mt:
	modal run -m modal_app.mt::upload_mt

modal-upload-speech:
	modal run -m modal_app.asr::upload_speech

modal-upload-llm:
	modal run -m modal_app.llm::upload_llm

modal-upload-punctuation:
	modal run -m modal_app.punctuation::upload_punctuation

# Out of $(UPLOADS) on purpose: 398 MB that moves once, for an account whose checkpoint
# volume does not already hold the encoder the punctuation heads sit on.
modal-upload-encoder:
	modal run -m modal_app.punctuation::upload_encoder

modal-upload-tts:
	modal run -m modal_app.tts::upload_tts

modal-train: ## Deploy, start the run, and tail it. COMPILE=1 RESUME=best START=n STEPS=n FORCE=1
	$(call deploy,train)
	$(PY) -m modal_app.launch $(if $(FORCE),--force,) $(if $(COMPILE),--compile,) \
		$(if $(STEPS),--steps $(STEPS),) $(if $(START),--schedule-start $(START),) \
		$(if $(RESUME),--resume-from $(RESUME),)

modal-logs: ## Follow ONE call's logs, never the whole app. FUNCTION=matoub_train|asr_train|...
	$(PY) -m modal_app.launch --logs $(if $(FUNCTION),--function $(FUNCTION),) \
		$(if $(CALL),--call-id $(CALL),)

modal-status: ## Is the run alive? Reports the call's state without attaching. FUNCTION=sweep
	$(PY) -m modal_app.launch --status $(if $(FUNCTION),--function $(FUNCTION),)

modal-cancel: ## Stop the run and the deployed app, so the next launch starts clean. FUNCTION=sweep
	$(PY) -m modal_app.launch --cancel $(if $(FUNCTION),--function $(FUNCTION),)

modal-smoke: ## 20 steps on a real GPU. COMPILE=1 also runs it compiled and prints the ratio
	modal run -m modal_app.train::smoke $(if $(COMPILE),--compare-compile,)

modal-infer: ## Fill a [MASK] with the run's best checkpoint. TEXT="... [MASK] ..."
	modal run -m modal_app.infer::fill_mask $(if $(TEXT),--text "$(TEXT)",)

modal-llm: modal-llm-$(or $(TASK),baseline) ## Phase 11 on GPU. TASK=baseline

modal-jugurtha: modal-jugurtha-$(or $(TASK),train) ## Phase 11 CPT. TASK=pack|train (spawned). EPOCHS STEPS RUN FORCE

modal-jugurtha-pack:
	modal run -m modal_app.jugurtha::pack $(if $(FORCE),--force,)

modal-jugurtha-train:
	$(call deploy,jugurtha)
	$(PY) -m modal_app.launch --function jugurtha_train $(if $(FORCE),--force,) \
		$(if $(EPOCHS),--epochs $(EPOCHS),) $(if $(STEPS),--steps $(STEPS),) \
		$(if $(RUN),--run-name $(RUN),)

modal-llm-baseline:
	modal run -m modal_app.llm::run_baseline $(if $(MODEL),--model $(MODEL),) \
		$(if $(DIRECTIONS),--directions $(DIRECTIONS),) \
		$(if $(SHOTS),--shot-count $(SHOTS),) $(if $(SENTENCES),--sentences $(SENTENCES),)

modal-asr-train: ## Phase 5 — deploy, spawn Fadhma, tail it. RUN EPOCHS STEPS FLOOR FORCE. FETCH=1 fetches and repacks first
	$(call deploy,asr)
	$(PY) -m modal_app.launch --function $(if $(FETCH),asr_pipeline,asr_finetune) \
		$(if $(RUN),--run-name $(RUN),) \
		$(if $(EPOCHS),--epochs $(EPOCHS),) $(if $(STEPS),--steps $(STEPS),) \
		$(if $(FLOOR),--floor $(FLOOR),) $(if $(FORCE),--force,)

modal-asr-repack: ## Phase 5 — deploy, spawn the CPU-only audio repack, tail it. THREADS SPLITS FORCE
	$(call deploy,asr)
	$(PY) -m modal_app.launch --function asr_repack \
		$(if $(THREADS),--threads $(THREADS),) $(if $(SPLITS),--splits $(SPLITS),) \
		$(if $(FORCE),--force,)

modal-asr: ## Phase 5 — Fadhma short jobs, attached. TASK=fetch|repack|lm|finetune|evaluate. RUN SPLIT STEPS LIMIT FORCE
	$(call deploy,asr)
	modal run -m modal_app.asr::run_asr $(if $(TASK),--task $(TASK),) $(if $(RUN),--run $(RUN),) \
		$(if $(SPLIT),--split $(SPLIT),) $(if $(STEPS),--max-steps $(STEPS),) \
		$(if $(LIMIT),--limit $(LIMIT),) $(if $(FORCE),--force,)

modal-tts: modal-tts-$(or $(TASK),baseline) ## Phase 12. TASK=baseline (12.1)|voices (12.3)|corpus (12.4, spawned; FORCE=1 rebuilds)|pull. LIMIT=20 smokes each

modal-tts-baseline:
	modal run -m modal_app.tts::run_tts $(if $(RUN),--run $(RUN),) $(if $(LIMIT),--limit $(LIMIT),)

modal-tts-voices:
	modal run -m modal_app.tts::run_voices $(if $(TOP),--top $(TOP),) $(if $(LIMIT),--limit $(LIMIT),)

modal-tts-corpus:
	$(call deploy,tts)
	$(PY) -m modal_app.launch --function tts_corpus \
		$(if $(RUN),--run-name $(RUN),) $(if $(LIMIT),--limit $(LIMIT),) \
		$(if $(FORCE),--force,)

modal-tts-pull:
	modal run -m modal_app.tts::pull_corpus $(if $(LIMIT),--limit $(LIMIT),)

modal-matoub: modal-matoub-$(or $(TASK),prepare) ## Task 12.6 — Kokoro fine-tune. TASK=prepare (CPU, no GPU)|stage1|stage2. ARM=restored|raw VOICE=kab_male|kab_female (stage2) LIMIT=<n> smokes EPOCHS FROM=<stage1 epoch> MAXLEN=<frames, the OOM lever> BATCH=<examples/step> FORCE
	@:

modal-matoub-prepare:
	modal run -m modal_app.matoub::matoub_prepare $(if $(ARM),--arm $(ARM),) \
		$(if $(LIMIT),--limit $(LIMIT),) $(if $(FORCE),--force,)

modal-matoub-stage1:
	$(call deploy,matoub)
	$(PY) -m modal_app.launch --function matoub_train --stage stage1 \
		$(if $(ARM),--arm $(ARM),) $(if $(EPOCHS),--epochs $(EPOCHS),) \
		$(if $(MAXLEN),--max-len $(MAXLEN),) $(if $(BATCH),--batch $(BATCH),) \
		$(if $(LIMIT),--limit $(LIMIT),) $(if $(RUN),--run-name $(RUN),)

modal-matoub-stage2:
	$(call deploy,matoub)
	$(PY) -m modal_app.launch --function matoub_train --stage stage2 \
		$(if $(VOICE),--voice $(VOICE),) \
		$(if $(ARM),--arm $(ARM),) $(if $(EPOCHS),--epochs $(EPOCHS),) \
		$(if $(FROM),--first-stage-epoch $(FROM),) $(if $(MAXLEN),--max-len $(MAXLEN),) \
		$(if $(BATCH),--batch $(BATCH),) \
		$(if $(LIMIT),--limit $(LIMIT),) $(if $(RUN),--run-name $(RUN),)

modal-tifinagh: ## Juba-27M on GPU — deploy, spawn, tail. TASK=train|evaluate. RUN STEPS SPLIT LIMIT FORCE
	$(call deploy,tifinagh)
	$(PY) -m modal_app.launch --function tifinagh_$(or $(TASK),train) \
		$(if $(RUN),--run-name $(RUN),) $(if $(STEPS),--steps $(STEPS),) \
		$(if $(SPLIT),--split $(SPLIT),) $(if $(LIMIT),--limit $(LIMIT),) \
		$(if $(FORCE),--force,)

modal-punctuation: ## Punctuation and casing on GPU — deploy, spawn, tail. TASK=train|evaluate. RUN EPOCHS SPLIT CHECKPOINT FORCE
	$(call deploy,punctuation)
	$(PY) -m modal_app.launch --function punctuation_$(or $(TASK),train) \
		$(if $(RUN),--run-name $(RUN),) $(if $(EPOCHS),--epochs $(EPOCHS),) \
		$(if $(SPLIT),--split $(SPLIT),) $(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),) \
		$(if $(FORCE),--force,)

modal-sentiment: ## Phase 7 sentiment on GPU. TASK=benchmark scores the encoder; TASK=build relabels from Tatoeba
	$(call deploy,sentiment)
	modal run -m modal_app.sentiment::run_sentiment --task $(or $(TASK),benchmark)

modal-translate-doc: ## Translate a long document detached. FILE=doc.txt DIR=eng-kab. Then modal-translate-pull
	$(call deploy,translate)
	$(PY) -m modal_app.launch --function mt_predict --file $(FILE) \
		$(if $(DIR),--direction $(DIR),)

modal-translate-pull: ## Copy finished translations off the volume into artifacts/translations/
	modal run -m modal_app.translate::pull_translations

modal-translate: ## Translate with the MT fine-tune. TEXT="..." or FILE=doc.txt OUT=path. DIR=eng-kab COMPARE=1
	modal run -m modal_app.translate::translate $(if $(TEXT),--text "$(TEXT)",) \
		$(if $(FILE),--file $(FILE),) $(if $(OUT),--out $(OUT),) \
		$(if $(DIR),--direction $(DIR),) $(if $(COMPARE),--compare,) \
		$(if $(WEIGHTS),--weights $(WEIGHTS),)

modal-bench-mt: ## Task 7.7 — score the MT baselines. WEIGHTS=<path> MODELS=<repo> DIRECTIONS=a,b
	modal run -m modal_app.bench::score_mt $(if $(MODELS),--models $(MODELS),) \
		$(if $(WEIGHTS),--weights $(WEIGHTS),) $(if $(DIRECTIONS),--directions $(DIRECTIONS),)

modal-synth: ## Generate kab bitext for new languages by pivot, detached. TARGETS="a b" TWO=1 LIMIT=n
	modal run -m modal_app.synth::upload_pivot
	$(call deploy,synth)
	$(PY) -m modal_app.launch --function synthesise \
		$(if $(TARGETS),--targets $(TARGETS),) $(if $(TWO),--two-teacher-only,) \
		$(if $(LIMIT),--steps $(LIMIT),)

modal-mt-train: ## Trim then fine-tune NLLB-1.3B, detached. SMALL=1 FREEZE=1 STEPS=n RUN=name FORCE=1
	modal run -m modal_app.mt::prepare $(if $(SMALL),--small,) $(if $(FORCE),--force,)
	$(call deploy,mt)
	$(PY) -m modal_app.launch --function finetune \
		$(if $(SMALL),--small,) $(if $(FREEZE),--freeze,) $(if $(STEPS),--steps $(STEPS),) \
		$(if $(RUN),--run-name $(RUN),)

.PHONY: $(shell grep -hoE '^[a-zA-Z_-]+:' $(MAKEFILE_LIST) | tr -d ':')
