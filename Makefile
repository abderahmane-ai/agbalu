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
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

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

embed: ## SiMohand sentence embeddings. TASK=coverage|corpus
	$(call cli,embed) $(or $(TASK),coverage)

# Training is not here: nothing in this project trains locally. `make modal-ocr` is the
# path, and `evaluate` is what produces the number the card carries.
ocr: ## Feraoun-36M OCR, GPU-free. TASK=generate|evaluate|infer (IMAGE=path)|transcribe-book (BOOK=dir). INPUT OUTPUT LINES BATCH RATIO DEVICE CHECKPOINT PAGE=1
	$(call cli,ocr) $(or $(TASK),evaluate) \
		$(if $(INPUT),--input $(INPUT),) $(if $(OUTPUT),--output $(OUTPUT),) \
		$(if $(LINES),--lines $(LINES),) $(if $(BATCH),--batch-size $(BATCH),) \
		$(if $(RATIO),--tifinagh-ratio $(RATIO),) $(if $(DEVICE),--device $(DEVICE),) \
		$(if $(IMAGE),--image $(IMAGE),) $(if $(PAGE),--page,) \
		$(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),) \
		$(if $(BOOK),--book-dir $(BOOK),)

standardise: ## Boulifa-48M, GPU-free. TASK=standardise TEXT="achimi..."|evaluate. LIMIT BATCH CHECKPOINT
	$(call cli,standardise) $(or $(TASK),evaluate) \
		$(if $(TEXT),"$(TEXT)",) $(if $(LIMIT),--limit $(LIMIT),) \
		$(if $(BATCH),--batch-size $(BATCH),) \
		$(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),)

RELEASE ?= artifacts/release

CHECKPOINTS ?= artifacts/checkpoints

# Datasets go through one path, because the path is where the checks are: the Hub validates
# card metadata on *render*, so a bad `task_categories` or `task_ids` publishes silently and
# is visible only to whoever opens the page. Models are staged per repo, since each has a
# different weight format.
DATASET_REPOS := bench lex sentiment inflect tifinagh punct g2p
MODEL_REPOS := encoder tokenizer mt juba
REPOS := $(MODEL_REPOS) $(DATASET_REPOS) org

# Out of the default set on purpose: each needs a checkpoint pulled off a volume first, so
# `make release` with no REPO would fail on whichever one this machine has not pulled.
PULLED_REPOS := fadhma belaid boulifa matoub simohand feraoun kabstandard

release: $(addprefix release-,$(or $(REPO),$(REPOS))) ## Stage repos under artifacts/release. REPO=encoder|tokenizer|mt|juba|bench|lex|sentiment|inflect|tifinagh|punct|g2p|org, or fadhma|belaid|boulifa|matoub|simohand|feraoun|kabstandard once pulled
	@echo 'staged under $(RELEASE) — review, then push each with: hf upload <repo> <dir> .'

$(addprefix release-,$(DATASET_REPOS)): release-%:
	$(PY) -m tools.export_datasets --only $* --out $(RELEASE)

release-org:
	@mkdir -p $(RELEASE)/org-README
	@cp docs/cards/organization.md $(RELEASE)/org-README/README.md
	@echo "Staged $(RELEASE)/org-README/README.md"

# `hf upload` can never update a Space — it calls `api/repos/create` unconditionally and
# 402s even against an existing public one — so the org card goes through `upload_file`,
# which does not re-create. MESSAGE is a variable because a commit message describing one
# release is wrong for the next.
ORG_MESSAGE ?= Update organization README
push-org: release-org ## Upload the organization card to the agbalu Space. MESSAGE="..."
	@$(PY) -c "\
from huggingface_hub import HfApi;\
print('Uploaded:', HfApi().upload_file(\
    path_or_fileobj='$(RELEASE)/org-README/README.md',\
    path_in_repo='README.md',\
    repo_id='agbalu/README',\
    repo_type='space',\
    commit_message='$(ORG_MESSAGE)',\
))"

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

# SiMohand is a native sentence-transformers model — no custom architecture, no stage_hub.
# The weights are already on disk from `modal volume get`; only the card needs syncing.
# The name repeats because `modal volume get` recreates the remote directory inside the
# local target, putting the model at `artifacts/simohand-base-v1/simohand-base-v1/`.
SIMOHAND_DIR ?= artifacts/simohand-base-v1/simohand-base-v1
release-simohand:
	@test -f $(SIMOHAND_DIR)/config.json || \
		{ echo "no model in $(SIMOHAND_DIR) — pull it first"; exit 1; }
	@cp docs/cards/simohand-278m.md $(SIMOHAND_DIR)/README.md
	@echo "$(SIMOHAND_DIR): $$(ls $(SIMOHAND_DIR) | wc -l | tr -d ' ') files"

# Weights arrive from `make modal-boulifa TASK=pull`. `export_checkpoint` strips the
# optimizer state, untangles the tied weights and writes model.safetensors + config.json +
# README.md; `stage_hub` adds the modelling code and refuses a directory that will not load
# back — the same two steps as Juba-27M.
release-boulifa:
	$(PY) -m tools.export_checkpoint \
		--source artifacts/boulifa/boulifa_best.pt \
		--out $(RELEASE)/Boulifa-48M \
		--card docs/cards/boulifa-48m.md
	$(PY) -m tools.stage_hub --repo boulifa --dir $(RELEASE)/Boulifa-48M

# `pos_encoder.pe` stays in the export: it is a registered buffer, so `from_pretrained`
# allocates it empty and fills it from the file. Dropped as derived it would come back as
# uninitialised memory and rotate every position by a garbage angle without raising.
FERAOUN_RUN ?= feraoun-36m-v1
release-feraoun:
	$(PY) -m tools.export_checkpoint \
		--source artifacts/runs/$(FERAOUN_RUN)/best.pt \
		--out $(RELEASE)/Feraoun-36M \
		--card docs/cards/feraoun-36m.md \
		--extra data/processed/bench/feraoun-v1-heldout.json
	$(PY) -m tools.stage_hub --repo feraoun --dir $(RELEASE)/Feraoun-36M

release-kabstandard:
	@mkdir -p $(RELEASE)/KabStandard
	@cp data/kabstandard/train.jsonl data/kabstandard/dev.jsonl data/kabstandard/test.jsonl \
		$(RELEASE)/KabStandard/
	@cp docs/cards/kabstandard.md $(RELEASE)/KabStandard/README.md
	@echo "$(RELEASE)/KabStandard: $$(ls $(RELEASE)/KabStandard | wc -l | tr -d ' ') files"

MATOUB_EPOCH ?= epoch_2nd_00003
modal-matoub-pull:
	@mkdir -p artifacts/matoub
	modal volume get --force agbalu-checkpoints \
		/tts/matoub/logs/kab_male/$(MATOUB_EPOCH).pth artifacts/matoub/$(MATOUB_EPOCH).pth

# The one release that ships a training checkpoint rather than an export, because StyleTTS2
# has no `save_pretrained` and its loader keys on the recipe's own component names. The card
# says so: the file carries optimizer state and is larger than the weights alone.
release-matoub:
	@mkdir -p $(RELEASE)/Matoub-82M
	@cp artifacts/matoub/$(MATOUB_EPOCH).pth $(RELEASE)/Matoub-82M/$(MATOUB_EPOCH).pth
	@cp src/agbalu/hub/matoub/inference.py $(RELEASE)/Matoub-82M/inference.py
	@cp docs/cards/matoub-82m.md $(RELEASE)/Matoub-82M/README.md
	@echo "$(RELEASE)/Matoub-82M: $$(ls $(RELEASE)/Matoub-82M | wc -l | tr -d ' ') files"

# One push target with a variable, not one per repository. The table is the only place a
# staged directory is paired with the repository it belongs in; two copies of that pairing
# is how a directory gets published under the wrong name.
CARDS := docs/cards
HF_bench       := datasets agbalu/KabBench       $(RELEASE)/KabBench      $(CARDS)/kabbench.md
HF_lex         := datasets agbalu/KabLex         $(RELEASE)/KabLex        $(CARDS)/kablex.md
HF_sentiment   := datasets agbalu/KabSentiment   $(RELEASE)/KabSentiment  $(CARDS)/kabsentiment.md
HF_inflect     := datasets agbalu/KabInflect     $(RELEASE)/KabInflect    $(CARDS)/kabinflect.md
HF_tifinagh    := datasets agbalu/KabTifinagh    $(RELEASE)/KabTifinagh   $(CARDS)/kabtifinagh.md
HF_punct       := datasets agbalu/KabPunct       $(RELEASE)/KabPunct      $(CARDS)/kabpunct.md
HF_g2p         := datasets agbalu/KabG2P         $(RELEASE)/KabG2P        $(CARDS)/kabg2p.md
HF_kabstandard := datasets agbalu/KabStandard    $(RELEASE)/KabStandard   $(CARDS)/kabstandard.md
HF_encoder     := models   agbalu/Masinissa-31M  $(RELEASE)/Masinissa-31M $(CARDS)/masinissa-31m.md
HF_tokenizer   := models   agbalu/Mammeri-Tok    $(RELEASE)/Mammeri-Tok   $(CARDS)/mammeri-tok.md
HF_mt          := models   agbalu/Amrouche-1.3B  $(RELEASE)/Amrouche-1.3B $(CARDS)/amrouche-1.3b.md
HF_juba        := models   agbalu/Juba-27M       $(RELEASE)/Juba-27M      $(CARDS)/juba-27m.md
HF_fadhma      := models   agbalu/Fadhma-300M    $(RELEASE)/Fadhma-300M   $(CARDS)/fadhma-300m.md
HF_belaid      := models   agbalu/Belaid-31M     $(RELEASE)/Belaid-31M    $(CARDS)/belaid-31m.md
HF_boulifa     := models   agbalu/Boulifa-48M    $(RELEASE)/Boulifa-48M   $(CARDS)/boulifa-48m.md
HF_matoub      := models   agbalu/Matoub-82M     $(RELEASE)/Matoub-82M    $(CARDS)/matoub-82m.md
HF_feraoun     := models   agbalu/Feraoun-36M    $(RELEASE)/Feraoun-36M   $(CARDS)/feraoun-36m.md
HF_simohand    := models   agbalu/SiMohand-278M  $(SIMOHAND_DIR)          $(CARDS)/simohand-278m.md

ALL_REPOS := bench lex sentiment inflect tifinagh punct g2p kabstandard encoder tokenizer \
	mt juba fadhma belaid boulifa matoub feraoun simohand

# Restaged first, always: `artifacts/release/` is git-ignored scratch and goes stale
# silently, so a push from whatever is already there can un-publish correct live text.
push: ## Restage and upload one repo. REPO=<any name `make release` takes>. Needs `hf auth login`
	@test -n "$(REPO)" || { echo "REPO= is required, e.g. make push REPO=feraoun"; exit 1; }
	@test -n "$(HF_$(REPO))" || { echo "no Hub repository registered for REPO=$(REPO)"; exit 1; }
	$(MAKE) release-$(REPO)
	hf upload $(word 2,$(HF_$(REPO))) $(word 3,$(HF_$(REPO))) . \
		--repo-type=$(patsubst %s,%,$(word 1,$(HF_$(REPO))))
	@echo "pushed $(word 2,$(HF_$(REPO))) — now diff the deployed card against docs/cards/"

# One file per repository, straight from `docs/cards/`. A card change needs no restaging and
# no weights: `make push REPO=mt` would re-upload 4.65 GB to correct a paragraph. Every
# upload is followed by reading the deployed file back and diffing it against its source,
# because a push that silently kept the old text looks exactly like a push that worked.
# The org profile is a Space, and `hf upload` cannot update one — it calls `repos/create`
# unconditionally and 402s against an existing Space — so it goes through `push-org`, which
# uses `upload_file`. Included here only for a full run, not for a single `REPO=`.
push-cards: $(addprefix push-card-,$(or $(REPO),$(ALL_REPOS))) ## Upload every card as its repo's README.md and verify each. REPO=<one name> for one
ifeq ($(REPO),)
	@$(MAKE) --no-print-directory push-org
endif
	@echo "every card uploaded and read back identical to docs/cards/"

hub_url = https://huggingface.co/$(if $(filter datasets,$(word 1,$(HF_$(1)))),datasets/,)$(word 2,$(HF_$(1)))

push-card-%:
	@test -f $(word 4,$(HF_$*)) || { echo "no card at $(word 4,$(HF_$*))"; exit 1; }
	@hf upload $(word 2,$(HF_$*)) $(word 4,$(HF_$*)) README.md \
		--repo-type=$(patsubst %s,%,$(word 1,$(HF_$*))) >/dev/null
	@curl -sfL $(call hub_url,$*)/resolve/main/README.md | diff -q - $(word 4,$(HF_$*)) >/dev/null \
		&& echo "  $(word 2,$(HF_$*)) matches $(word 4,$(HF_$*))" \
		|| { echo "  MISMATCH: $(word 2,$(HF_$*)) does not match $(word 4,$(HF_$*))"; exit 1; }

infer-matoub: ## Synthesise, pull the wav and play it. TEXT="Azul..." VOICE STAGE CHECKPOINT OUT
	$(PY) -m tools.infer_matoub $(if $(TEXT),--text "$(TEXT)",) $(if $(VOICE),--voice "$(VOICE)",) \
		$(if $(STAGE),--stage "$(STAGE)",) $(if $(CHECKPOINT),--checkpoint "$(CHECKPOINT)",) \
		$(if $(OUT),--out "$(OUT)",)

modal-boulifa: modal-boulifa-$(or $(TASK),train) ## Boulifa-48M. TASK=prepare (CPU, builds KabStandard)|train (spawned)|pull. EPOCHS BATCH LIMIT

modal-boulifa-prepare:
	modal run -m modal_app.boulifa::boulifa_prepare $(if $(LIMIT),--limit $(LIMIT),)

modal-boulifa-train:
	$(call deploy,boulifa)
	$(PY) -m modal_app.launch --function boulifa_train \
		$(if $(EPOCHS),--epochs $(EPOCHS),) $(if $(BATCH),--batch $(BATCH),) \
		$(if $(LIMIT),--limit $(LIMIT),)

modal-boulifa-pull:
	@mkdir -p artifacts/boulifa
	modal volume get --force agbalu-checkpoints \
		/boulifa/boulifa_best.pt artifacts/boulifa/boulifa_best.pt

UPLOADS := corpus bench mt speech llm tts punctuation ocr embed

modal-upload: $(addprefix modal-upload-,$(or $(TASK),$(UPLOADS))) ## Push what a container reads off the volumes. TASK=corpus|bench|mt|speech|llm|tts|punctuation|ocr|embed|encoder. Tifinagh fetches its own

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

modal-upload-embed:
	modal run -m modal_app.simohand::upload_embed

modal-upload-ocr:
	modal run -m modal_app.ocr::upload_ocr

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

modal-matoub: modal-matoub-$(or $(TASK),prepare) ## Task 12.6 — Kokoro fine-tune. TASK=prepare (CPU, no GPU)|stage1|stage2|infer|pull. ARM=restored|raw VOICE=kab_male|kab_female (stage2) LIMIT=<n> smokes EPOCHS FROM=<stage1 epoch> MAXLEN=<frames, the OOM lever> BATCH=<examples/step> FORCE

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

modal-matoub-infer:
	modal run -m modal_app.matoub::matoub_infer $(if $(TEXT),--text "$(TEXT)",) \
		$(if $(VOICE),--voice $(VOICE),) $(if $(ARM),--arm $(ARM),) \
		$(if $(STAGE),--stage $(STAGE),) $(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),) \
		$(if $(LIMIT),--limit $(LIMIT),)

modal-simohand: ## SiMohand sentence embeddings on Modal. TASK=prepare|train|eval. RUN EPOCHS STEPS LIMIT BATCH FORCE
	$(call deploy,simohand)
	$(PY) -m modal_app.launch --function simohand_$(or $(TASK),train) \
		$(if $(RUN),--run-name $(RUN),) $(if $(EPOCHS),--epochs $(EPOCHS),) \
		$(if $(STEPS),--steps $(STEPS),) $(if $(LIMIT),--limit $(LIMIT),) \
		$(if $(BATCH),--batch $(BATCH),) $(if $(FORCE),--force,)

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

# RATIO is the dual-script lever: 0.0 trains Latin only, 0.50 the 50/50 Latin+Tifinagh
# continuation the release was cut from. Every default lives in `_feraoun_kwargs`, so there
# is no second target restating them.
modal-ocr: ## Feraoun-36M on GPU — deploy, spawn, tail. TASK=train|smoke. RUN EPOCHS BATCH LR LINES RESUME RATIO
	$(call deploy,ocr)
	$(PY) -m modal_app.launch --function feraoun_$(or $(TASK),train) \
		$(if $(RUN),--run-name $(RUN),) $(if $(EPOCHS),--epochs $(EPOCHS),) \
		$(if $(BATCH),--batch-size $(BATCH),) $(if $(LR),--lr $(LR),) \
		$(if $(LINES),--max-lines $(LINES),) $(if $(RESUME),--resume-from $(RESUME),) \
		$(if $(RATIO),--tifinagh-ratio $(RATIO),)

# `modal volume get`, as every other pull here does: a 415 MB checkpoint does not travel
# through a function's return value, and `modal run`'s stdout carries the log stream.
modal-ocr-pull: ## Copy the Feraoun checkpoint off the volume. RUN=feraoun-36m-v1
	@mkdir -p artifacts/runs/$(or $(RUN),$(FERAOUN_RUN))
	modal volume get --force agbalu-checkpoints \
		/feraoun/$(or $(RUN),$(FERAOUN_RUN))/best.pt \
		artifacts/runs/$(or $(RUN),$(FERAOUN_RUN))/best.pt

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
