# How the two Claudes talk

Two Claudes work on this repo: Claude-A on SADIES COMPUTER and Claude-B on CHELSEAS COMPUTER. Git is the only shared channel, so messages are files.

## Files

- `comms/claude_A.md`: written **only** by Claude-A (SADIES COMPUTER, owns the data and the run).
- `comms/claude_B.md`: written **only** by Claude-B (CHELSEAS COMPUTER).
- Each Claude **reads** the other's file and never edits it. Because each file has one writer, they never conflict in Git.

## Message format (append to the bottom of your own file; never edit or delete an earlier message)

```
## [N] 2026-10-03 22:10 ET | from A | to B | needs reply: yes/no | topic: short title
Body: what you need, found or did. Plain text. Name files and commit hashes. Never include the API key, any
credential, or licensed data (prices, bars, raw responses).
```

Number your own messages 1, 2, 3 in order. Reply by writing a new message that starts with "Re: A#3" or "Re: B#2".
Mark a request "needs reply: yes" only when you cannot continue without an answer.

## Syncing (the humans push and pull; Claudes only commit)

1. After writing a message, **commit** it (`git add comms/claude_X.md && git commit -m "comms: ..."`).
2. Ask the human to push; they pull on the other computer. Do not push yourself unless the human explicitly asks.
3. Before you start work and every time you are idle, ask the human to `git pull`, then read the other file's new messages.
4. If the human says "check comms", read the other file first.

## Rules for what you may ask each other

- Requests to change a file you do not own go here first. The owner decides.
- Anything touching the hypothesis, the test plan or the data window needs the humans' approval, not just Claude's.
- Results from the real run are produced only by Claude-A; Claude-B never runs the test on real data.
- Both Claudes follow `docs/HANDOFF.md` hard rules without exception. The key must never appear in these files.

## Status board (Claude-A updates this block at the top of its own file)

See the top of `comms/claude_A.md`.
