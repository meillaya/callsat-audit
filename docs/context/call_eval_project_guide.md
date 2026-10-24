Title: call-eval — Jev Project Guide · JevList

URL Source: https://jevlist.ai/projects/call-eval

Markdown Content:
[Skip to content](https://jevlist.ai/projects/call-eval#main)

[Jev List](https://jevlist.ai/)[Directory](https://jevlist.ai/)[Resources](https://jevlist.ai/resources)[Knowledge](https://jevlist.ai/knowledge)[About](https://jevlist.ai/about)

[](https://github.com/AppitStudio/awesome-jev?ref=jevlist.ai "Awesome Jev on GitHub")[Submit](https://jevlist.ai/submit)

*   [Directory](https://jevlist.ai/)
*   [Resources](https://jevlist.ai/resources)
*   [Knowledge](https://jevlist.ai/knowledge)
*   [About](https://jevlist.ai/about)
*   [Submit a project](https://jevlist.ai/submit)

[Directory](https://jevlist.ai/)/[Customer feedback and marketing](https://jevlist.ai/categories/customer-feedback-and-marketing)/call-eval

 C 

Customer feedback and marketing

# call-eval

Post-call customer-experience evaluation for contact-centre transcripts: Jev (or an offline rule extractor) pulls typed facts with evidence spans, deterministic code turns them into satisfaction and sentiment scores, and a calibration layer anchors scores to surveys — all on synthetic data with a sealed holdout.

Python CLI package + static demo site (MIT)#calibration#contact-center#llm-evaluation#python

[Open the app](https://henryzhangpku.github.io/call-eval/?ref=jevlist.ai)[View on GitHub](https://github.com/henryzhangpku/call-eval?ref=jevlist.ai) 0 Python MIT pushed 1w ago

0 views Reviewed Oct 7, 2026 Pinned to abba9792be Synced 36 seconds ago[Edit this page on GitHub](https://github.com/AppitStudio/awesome-jev/blob/main/community/projects/tools/call-eval.md?ref=jevlist.ai)

On this page
1.   [When to use](https://jevlist.ai/projects/call-eval#when-to-use)
2.   [How it works](https://jevlist.ai/projects/call-eval#how-it-works)
3.   [Get started](https://jevlist.ai/projects/call-eval#get-started)
4.   [Examples and demos](https://jevlist.ai/projects/call-eval#examples-and-demos)
5.   [Limits and data handling](https://jevlist.ai/projects/call-eval#limits-and-data-handling)
6.   [Review and maintenance](https://jevlist.ai/projects/call-eval#review-and-maintenance)

## [When to use](https://jevlist.ai/projects/call-eval#when-to-use)

*   Prototype auditable contact-centre QA where the model extracts facts and code does the scoring.
*   Compare a rule-based extractor with Jev on the same synthetic calls, dev split and sealed holdout.
*   Mismatch: all data is synthetic (400 generated calls, simulated annotators and surveys); numbers do not describe real callers.

## [How it works](https://jevlist.ai/projects/call-eval#how-it-works)

[`calleval/extract/jev.py`](https://github.com/henryzhangpku/call-eval/blob/abba9792be56f87c2512e390a7916bdeeb112ab6/calleval/extract/jev.py?ref=jevlist.ai) asks Jev typed questions per call segment (was each request resolved, caller effort, stated feeling) with evidence locations; [`calleval/scoring.py`](https://github.com/henryzhangpku/call-eval/blob/abba9792be56f87c2512e390a7916bdeeb112ab6/calleval/scoring.py?ref=jevlist.ai) and [`calleval/calibration.py`](https://github.com/henryzhangpku/call-eval/blob/abba9792be56f87c2512e390a7916bdeeb112ab6/calleval/calibration.py?ref=jevlist.ai) turn those facts into scores and anchor them to survey answers. The Jev backend replays [`cache/jev_cache.jsonl`](https://github.com/henryzhangpku/call-eval/blob/abba9792be56f87c2512e390a7916bdeeb112ab6/cache/jev_cache.jsonl?ref=jevlist.ai) and calls the API only on a miss; design decisions are logged in [`DECISIONS.md`](https://github.com/henryzhangpku/call-eval/blob/abba9792be56f87c2512e390a7916bdeeb112ab6/DECISIONS.md?ref=jevlist.ai).

## [Get started](https://jevlist.ai/projects/call-eval#get-started)

Run end to end offline, then with Jev (cache first):

```sh
git clone https://github.com/henryzhangpku/call-eval.git && cd call-eval
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
python -m calleval run --backend offline   # no key, no network
python -m calleval run --backend jev       # replays cache; API only on a miss
python -m calleval evaluate
```

The offline backend and cached Jev runs make no API calls; cache misses are billed to your TypeSafe key.

## [Examples and demos](https://jevlist.ai/projects/call-eval#examples-and-demos)

*   Static demo site (GitHub Pages) linked as the product homepage; README _What it proves, measured_ (kappa vs human ceiling on dev and sealed holdout).
*   Methodology: [`METHODOLOGY.md`](https://github.com/henryzhangpku/call-eval/blob/abba9792be56f87c2512e390a7916bdeeb112ab6/METHODOLOGY.md?ref=jevlist.ai).

## [Limits and data handling](https://jevlist.ai/projects/call-eval#limits-and-data-handling)

Agreement numbers are the maintainer's, on synthetic template-generated calls (optimistic for any extractor); the offline rules were written against those templates. Transcript segments go to TypeSafe on live runs. Not run on the review host.

## [Review and maintenance](https://jevlist.ai/projects/call-eval#review-and-maintenance)

Reviewed **2026-10-07** (Europe/Sofia) at [commit abba9792be56](https://github.com/henryzhangpku/call-eval/tree/abba9792be56f87c2512e390a7916bdeeb112ab6?ref=jevlist.ai). Inspected the upstream README, LICENSE, and the Jev integration described there; install and live inference were not run on the review host. Catalog checks (`npm run lint`, `check:links`, `check:community`) ran locally; they verify navigation, not project claims. AI-assisted review.

At a glance

Source[Source](https://github.com/henryzhangpku/call-eval?ref=jevlist.ai)

Product homepage[henryzhangpku.github.io](https://henryzhangpku.github.io/call-eval/?ref=jevlist.ai)

Maintainer[henryzhangpku](https://github.com/henryzhangpku?ref=jevlist.ai). Independently curated.

Format Python package and CLI (`python -m calleval`), committed synthetic data, Jev response cache and static demo

Requirements Python 3 with `pip install -e .`; no key for the offline backend; `TYPESAFE_API_KEY` for live Jev calls on cache misses.

License[MIT](https://github.com/henryzhangpku/call-eval/blob/abba9792be56f87c2512e390a7916bdeeb112ab6/LICENSE?ref=jevlist.ai). External project keeps its own license.

Disclosure AI-assisted catalog review; no affiliation with the maintainer. Listing is not an endorsement. Install and live inference not run on the review host.

On this page

1.   [When to use](https://jevlist.ai/projects/call-eval#when-to-use)
2.   [How it works](https://jevlist.ai/projects/call-eval#how-it-works)
3.   [Get started](https://jevlist.ai/projects/call-eval#get-started)
4.   [Examples and demos](https://jevlist.ai/projects/call-eval#examples-and-demos)
5.   [Limits and data handling](https://jevlist.ai/projects/call-eval#limits-and-data-handling)
6.   [Review and maintenance](https://jevlist.ai/projects/call-eval#review-and-maintenance)

More in Customer feedback and marketing

*   [J jev-seo (AgriciDaniel)Live site SEO audit from one URL: crawl/rules/PageSpeed plus TypeSafe Jev judgments; PDF/XLSX/Markdown (distinct from Rust jev-seo).](https://jevlist.ai/projects/agrici-jev-seo)
*   [J jev-seo Local SEO/GEO CLI and MCP: DuckDuckGo SERP/audits plus optional TypeSafe Jev intent and visibility judgments.](https://jevlist.ai/projects/jev-seo)
*   [D discourse-chatbot The original Discourse AI Chatbot plugin; optional System One settings (default model jev-latest) let Jev decide whether a question belongs to the community and which tools the bot should use.](https://jevlist.ai/projects/discourse-chatbot)
*   [J jev-linkedin-saved-classifier LinkedIn saved-posts board classified into filters by TypeSafe Jev](https://jevlist.ai/projects/jev-linkedin-saved-classifier)

## Questions about call-eval

### What is call-eval?

Post-call customer-experience evaluation for contact-centre transcripts: Jev (or an offline rule extractor) pulls typed facts with evidence spans, deterministic code turns them into satisfaction and sentiment scores, and a calibration layer anchors scores to surveys — all on synthetic data with a sealed holdout.

### What does call-eval require?

Python 3 with pip install -e .; no key for the offline backend; TYPESAFE_API_KEY for live Jev calls on cache misses.

### When was the call-eval guide reviewed?

The guide records a review date of October 7, 2026. The reviewed revision is abba9792be56f87c2512e390a7916bdeeb112ab6. This date refers to the guide review, not the latest software release.

A growing Jev community

Visitors measured by [DataFast](https://datafa.st/?ref=jevlist.ai)

Since launch—

Last 24 hours—

Live visitors—

Live = active in the last 10 minutes. Totals refresh every 5 minutes; live every minute.

[Jev List](https://jevlist.ai/)
An independent directory of apps, libraries and examples built on Jev, TypeSafe AI’s typed-decision model. Every page is rendered from the open [Awesome Jev](https://github.com/AppitStudio/awesome-jev?ref=jevlist.ai) repository.

Not affiliated with or endorsed by TypeSafe AI. A listing is not an endorsement.

Categories

*   [Web apps](https://jevlist.ai/categories/web-apps)
*   [Desktop apps](https://jevlist.ai/categories/desktop-apps)
*   [macOS apps](https://jevlist.ai/categories/macos-apps)
*   [Android apps](https://jevlist.ai/categories/android-apps)
*   [Windows apps](https://jevlist.ai/categories/windows-apps)
*   [Browser extensions](https://jevlist.ai/categories/browser-extensions)
*   [Command-line apps](https://jevlist.ai/categories/command-line-apps)
*   [Discord bots](https://jevlist.ai/categories/discord-bots)
*   [Telegram bots](https://jevlist.ai/categories/telegram-bots)
*   [Browser and computer use](https://jevlist.ai/categories/browser-and-computer-use)
*   [Customer feedback and marketing](https://jevlist.ai/categories/customer-feedback-and-marketing)
*   [Developer tools](https://jevlist.ai/categories/developer-tools)
*   [Games and simulation](https://jevlist.ai/categories/games-and-simulation)
*   [Home automation](https://jevlist.ai/categories/home-automation)
*   [Independent model research](https://jevlist.ai/categories/independent-model-research)
*   [SDKs and integrations](https://jevlist.ai/categories/sdks-and-integrations)
*   [Search and retrieval](https://jevlist.ai/categories/search-and-retrieval)

Site

*   [Resources](https://jevlist.ai/resources)
*   [Knowledge base](https://jevlist.ai/knowledge)
*   [Submit a project](https://jevlist.ai/submit)
*   [About](https://jevlist.ai/about)
*   [Privacy](https://jevlist.ai/privacy-policy)
*   [Terms](https://jevlist.ai/terms-and-conditions)
*   Cookie preferences
*   [Sponsor](https://jevlist.ai/sponsor)
*   [Source on GitHub](https://github.com/AppitStudio/awesome-jev?ref=jevlist.ai)
*   [What is Jev?](https://docs.typesafe.ai/introduction?ref=jevlist.ai)
*   [llms.txt](https://jevlist.ai/llms.txt) · [RSS](https://jevlist.ai/feed.xml)

© 2026 [Appit Studio](https://appitstudio.com/?ref=jevlist.ai) · List content CC0, code MIT.

Built with Laravel · Livewire · Flux

[+Open box Your brand here ↗Launch price · 50% off~~€250~~**€125** / 30 days 10k+ visitors in the last 30 days.](https://jevlist.ai/sponsor/box/4/click)

[![Image 1](https://logos.jevlist.ai/jevlist/sponsors/q7y2VMFJip5QoHKaQcRMpetqfXm0rbBXdNU52toE.webp)E By Appit Studio ExtraDock Put a dock on every screen](https://jevlist.ai/sponsor/events/2/click?signature=b933ccb050819c6241e7b96a256cc1a9b005195bbae3cb389fd1ac17c24099c3)

[![Image 2](https://logos.jevlist.ai/jevlist/sponsors/VkeLVGe2xh7J0NjJTVyCny92u29WtQo4JkGs9GI5.webp)A By Appit Studio Appit Bundle The macOS experience reimagined](https://jevlist.ai/sponsor/events/1/click?signature=7a131838cf315c59ceb1b24da7539b6ac231181a03aa4c8f950878ec8e3debd5)

[![Image 3](https://logos.jevlist.ai/jevlist/sponsors/6ZbI8uHcuVusqwP3ZepIfrfWqlWoL7z0209KzOR7.webp)C By Appit Studio CoolDock A smart second dock full of useful widgets](https://jevlist.ai/sponsor/events/3/click?signature=34244f4f802be3ca678cc6357ccf0dafde630e33bae6eedee83ab4a433d5139c)

[+Open box Your brand here ↗Launch price · 50% off~~€250~~**€125** / 30 days 10k+ visitors in the last 30 days.](https://jevlist.ai/sponsor/box/5/click)

[+Open box Your brand here ↗Launch price · 50% off~~€250~~**€125** / 30 days 10k+ visitors in the last 30 days.](https://jevlist.ai/sponsor/box/6/click)

[+Open box Your brand here ↗Launch price · 50% off~~€250~~**€125** / 30 days 10k+ visitors in the last 30 days.](https://jevlist.ai/sponsor/box/4/click)

[![Image 4](https://logos.jevlist.ai/jevlist/sponsors/q7y2VMFJip5QoHKaQcRMpetqfXm0rbBXdNU52toE.webp)E By Appit Studio ExtraDock Put a dock on every screen](https://jevlist.ai/sponsor/events/2/click?signature=b933ccb050819c6241e7b96a256cc1a9b005195bbae3cb389fd1ac17c24099c3)

[![Image 5](https://logos.jevlist.ai/jevlist/sponsors/VkeLVGe2xh7J0NjJTVyCny92u29WtQo4JkGs9GI5.webp)A By Appit Studio Appit Bundle The macOS experience reimagined](https://jevlist.ai/sponsor/events/1/click?signature=7a131838cf315c59ceb1b24da7539b6ac231181a03aa4c8f950878ec8e3debd5)

[![Image 6](https://logos.jevlist.ai/jevlist/sponsors/6ZbI8uHcuVusqwP3ZepIfrfWqlWoL7z0209KzOR7.webp)C By Appit Studio CoolDock A smart second dock full of useful widgets](https://jevlist.ai/sponsor/events/3/click?signature=34244f4f802be3ca678cc6357ccf0dafde630e33bae6eedee83ab4a433d5139c)

[+Open box Your brand here ↗Launch price · 50% off~~€250~~**€125** / 30 days 10k+ visitors in the last 30 days.](https://jevlist.ai/sponsor/box/5/click)

[+Open box Your brand here ↗Launch price · 50% off~~€250~~**€125** / 30 days 10k+ visitors in the last 30 days.](https://jevlist.ai/sponsor/box/6/click)

[![Image 7](https://logos.jevlist.ai/jevlist/sponsors/6ZbI8uHcuVusqwP3ZepIfrfWqlWoL7z0209KzOR7.webp)C By Appit Studio CoolDock A smart second dock full of useful widgets](https://jevlist.ai/sponsor/events/3/click?signature=34244f4f802be3ca678cc6357ccf0dafde630e33bae6eedee83ab4a433d5139c)

[+Open box Your brand here ↗Launch price · 50% off~~€250~~**€125** / 30 days 10k+ visitors in the last 30 days.](https://jevlist.ai/sponsor/box/5/click)

[+Open box Your brand here ↗Launch price · 50% off~~€250~~**€125** / 30 days 10k+ visitors in the last 30 days.](https://jevlist.ai/sponsor/box/6/click)

[+Open box Your brand here ↗Launch price · 50% off~~€250~~**€125** / 30 days 10k+ visitors in the last 30 days.](https://jevlist.ai/sponsor/box/4/click)

[![Image 8](https://logos.jevlist.ai/jevlist/sponsors/q7y2VMFJip5QoHKaQcRMpetqfXm0rbBXdNU52toE.webp)E By Appit Studio ExtraDock Put a dock on every screen](https://jevlist.ai/sponsor/events/2/click?signature=b933ccb050819c6241e7b96a256cc1a9b005195bbae3cb389fd1ac17c24099c3)

[![Image 9](https://logos.jevlist.ai/jevlist/sponsors/VkeLVGe2xh7J0NjJTVyCny92u29WtQo4JkGs9GI5.webp)A By Appit Studio Appit Bundle The macOS experience reimagined](https://jevlist.ai/sponsor/events/1/click?signature=7a131838cf315c59ceb1b24da7539b6ac231181a03aa4c8f950878ec8e3debd5)

[![Image 10](https://logos.jevlist.ai/jevlist/sponsors/6ZbI8uHcuVusqwP3ZepIfrfWqlWoL7z0209KzOR7.webp)C By Appit Studio CoolDock A smart second dock full of useful widgets](https://jevlist.ai/sponsor/events/3/click?signature=34244f4f802be3ca678cc6357ccf0dafde630e33bae6eedee83ab4a433d5139c)

[+Open box Your brand here ↗Launch price · 50% off~~€250~~**€125** / 30 days 10k+ visitors in the last 30 days.](https://jevlist.ai/sponsor/box/5/click)

[+Open box Your brand here ↗Launch price · 50% off~~€250~~**€125** / 30 days 10k+ visitors in the last 30 days.](https://jevlist.ai/sponsor/box/6/click)

[+Open box Your brand here ↗Launch price · 50% off~~€250~~**€125** / 30 days 10k+ visitors in the last 30 days.](https://jevlist.ai/sponsor/box/4/click)

[![Image 11](https://logos.jevlist.ai/jevlist/sponsors/q7y2VMFJip5QoHKaQcRMpetqfXm0rbBXdNU52toE.webp)E By Appit Studio ExtraDock Put a dock on every screen](https://jevlist.ai/sponsor/events/2/click?signature=b933ccb050819c6241e7b96a256cc1a9b005195bbae3cb389fd1ac17c24099c3)

[![Image 12](https://logos.jevlist.ai/jevlist/sponsors/VkeLVGe2xh7J0NjJTVyCny92u29WtQo4JkGs9GI5.webp)A By Appit Studio Appit Bundle The macOS experience reimagined](https://jevlist.ai/sponsor/events/1/click?signature=7a131838cf315c59ceb1b24da7539b6ac231181a03aa4c8f950878ec8e3debd5)
