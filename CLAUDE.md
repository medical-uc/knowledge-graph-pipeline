# Project Instructions

## Project Overview

This project builds a staged pipeline that turns medical source material
(textbooks, notes, guidelines, including figures) into a queryable semantic-web
knowledge graph (RDF/OWL, queried with SPARQL). Read `DETAILS.md` to get a
comprehensive overview of the pipeline.

## Input Files

All the input files that will be fed to this pipeline is located in the `inputs`
directory.

## Rules (Do Not Break)

- All outputs should be put into the `artifacts` directory.
- Scripts should print to the console to indicate their progress. Do not use
  progress bars like `tqdm` because they can break on some terminals. Always use
  numbers that are batched; e.g., `10/100`, `20/100`, depending on how many
  items there are to process to indicate progress.
- Comments in the code should not address changes that I get from you. Write
  them as if the code's been like that from the start.
- Never make Git PRs on your own.

## Changelog Maintenance Rules

### Track Changes

Whenever you implement a new feature, fix a bug, change existing functionality,
or deprecate something, you must update the `CHANGELOG.md` file.

### Format Standard

Follow the [Keep a Changelog](https://keepachangelog.com/) standard.

### Structure

Always add entries under the `## [Unreleased]` section at the top of
`CHANGELOG.md` unless a formal version release is being cut.

Use appropriate subheadings: `### Added`, `### Changed`, `### Deprecated`,
`### Removed`, `### Fixed`, or `### Security`.

### Workflow Integration

Before finishing a task automatically update `CHANGELOG.md` to reflect the
modifications that were made.
