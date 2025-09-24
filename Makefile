.PHONY
enable-trace:
	export LANGSMITH_TRACING=true
	export LANGSMITH_API_KEY="TODO"

browser-setup:
	export USER_AGENT="RagdollAgent"