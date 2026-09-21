# Web Search Tools Available

## Current Status

| Tool | Works? | Notes |
|------|--------|-------|
| `WebSearch` | ✅ Yes | Returns link titles + URLs + summary. Good for finding documentation pages. |
| `WebFetch` | ✅ Yes | Fetches and extracts content from a URL. Handles redirects (re-invoke with redirect URL). Works with NVIDIA/docs.omniverse.nvidia.com domains. |
| `WebSearch` + custom API | ⚠️ May not work | The default Claude web search may fail when using custom API endpoints. Test per-session. |

## How to Use

### WebSearch
```
WebSearch(query="Isaac Sim Replicator generate fixed number of frames")
```
Returns: search result links with titles, URLs, and AI-summarized content.

### WebFetch
```
WebFetch(url="https://docs.isaacsim.omniverse.nvidia.com/...", prompt="extract code examples about X")
```
Returns: extracted content filtered by the prompt. Handles 301 redirects gracefully.

### Test Pattern
Always run a simple WebSearch first to verify the tool works in the current session before relying on it for research.
