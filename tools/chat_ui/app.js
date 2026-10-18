/**
 * ==============================================================================
 * AƔBALU: Jugurtha-4B Frontend Chat & Reasoning Engine
 * ==============================================================================
 */

const DEFAULT_SYSTEM_PROMPT = `Isem-ik d Jugurtha. D amallal aqbayli i d-yessnulfa Ainouche Abderahmane deg usenfar AƔBALU.
Tiririt-ik ad tili s teqbaylit (s usekkil alatini) s wudem amezwaru, anagar ma yella aseqdac yessuter-d tutlayt nniḍen neɣ tasuqilt.
Tarrayt n umeslay d teflest:
- Meslay akken i heddren yimdanen deg tmetti: s wawal aḥeqqani, s tseddi, s tefrit d leqder ameqran.
- Ur ttili ara d aɛlayan, ur ttɛanad ara medden, ur ssimɣur ara iman-ik yerna ur tt-ttarra ara d amassan fell-asen. Err-d s wannuz, s leḥnana, s leɛqel d tmusni tamedrart.
- Ssexdem tameslayt yelhan n yal ass i fehmen akk medden, bla asuffeɣ, bla aselket d wawalen ur nelli d ilaqiyen.
- Mi ara d-tefkeḍ tiririt, fferz igburen s tseddi. Ma yella d tamsalt n leḥsab neɣ d tamusni lqayen, smekti s teqbaylit s talwit uqbel ad d-tefkeḍ tiririt taneggarut.
- Ur d-ttader ara tikwulya nniḍen neɣ imesnulfuyen nniḍen; d Jugurtha i d isem-ik, d Ainouche Abderahmane i k-yessnulfan deg usenfar AƔBALU.`;

class JugurthaChatApp {
  constructor() {
    this.messages = [];
    this.abortController = null;
    this.isGenerating = false;

    // Load persisted settings or defaults
    this.settings = {
      endpoint: localStorage.getItem('jugurtha_endpoint') || '/v1',
      model: localStorage.getItem('jugurtha_model') || 'jugurtha-4b',
      temperature: parseFloat(localStorage.getItem('jugurtha_temp') || '0.3'),
      maxTokens: parseInt(localStorage.getItem('jugurtha_max_tokens') || '8192', 10) <= 1024 ? 8192 : parseInt(localStorage.getItem('jugurtha_max_tokens') || '8192', 10),
      systemPrompt: localStorage.getItem('jugurtha_sys_prompt') || DEFAULT_SYSTEM_PROMPT,
      enableThinking: localStorage.getItem('jugurtha_thinking') !== 'false'
    };

    this.initDOMElements();
    this.bindEvents();
    this.initSettingsUI();
    this.checkServerHealth();
    setInterval(() => this.checkServerHealth(), 10000);
  }

  initDOMElements() {
    this.dom = {
      thread: document.getElementById('chat-thread'),
      container: document.getElementById('messages-container'),
      welcomeHero: document.getElementById('welcome-hero'),
      userInput: document.getElementById('user-input'),
      sendBtn: document.getElementById('send-btn'),
      stopBtn: document.getElementById('stop-btn'),
      thinkingSwitch: document.getElementById('thinking-switch'),
      statusDot: document.getElementById('status-dot'),
      statusLabel: document.getElementById('status-label'),
      settingsModal: document.getElementById('settings-modal'),
      settingsTrigger: document.getElementById('settings-trigger-btn'),
      closeSettings: document.getElementById('close-settings-btn'),
      saveSettings: document.getElementById('save-settings-btn'),
      cancelSettings: document.getElementById('cancel-settings-btn'),
      resetSysPrompt: document.getElementById('reset-system-prompt'),
      newChatBtn: document.getElementById('new-chat-btn'),
      mobileMenuBtn: document.getElementById('mobile-menu-btn'),
      sidebar: document.getElementById('sidebar'),
      
      // Settings form
      apiEndpointInput: document.getElementById('api-endpoint'),
      modelAliasInput: document.getElementById('model-alias'),
      tempSlider: document.getElementById('temp-slider'),
      tempVal: document.getElementById('temp-val'),
      maxTokensInput: document.getElementById('max-tokens-input'),
      sysPromptInput: document.getElementById('system-prompt-input'),
    };
  }

  bindEvents() {
    // Textarea input auto-grow
    this.dom.userInput.addEventListener('input', () => {
      this.dom.userInput.style.height = 'auto';
      this.dom.userInput.style.height = Math.min(this.dom.userInput.scrollHeight, 180) + 'px';
    });

    // Enter / Shift+Enter handling
    this.dom.userInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        this.handleUserSend();
      }
    });

    this.dom.sendBtn.addEventListener('click', () => this.handleUserSend());
    this.dom.stopBtn.addEventListener('click', () => this.stopGeneration());
    this.dom.newChatBtn.addEventListener('click', () => this.startNewChat());

    // Prompt chips
    document.querySelectorAll('.prompt-chip').forEach(chip => {
      chip.addEventListener('click', () => {
        const prompt = chip.getAttribute('data-prompt');
        if (prompt) {
          this.dom.userInput.value = prompt;
          this.handleUserSend();
        }
      });
    });

    // Thinking mode switch
    this.dom.thinkingSwitch.checked = this.settings.enableThinking;
    this.dom.thinkingSwitch.addEventListener('change', (e) => {
      this.settings.enableThinking = e.target.checked;
      localStorage.setItem('jugurtha_thinking', this.settings.enableThinking);
    });

    // Settings Modal
    this.dom.settingsTrigger.addEventListener('click', () => this.openSettings());
    this.dom.closeSettings.addEventListener('click', () => this.closeSettings());
    this.dom.cancelSettings.addEventListener('click', () => this.closeSettings());
    this.dom.saveSettings.addEventListener('click', () => this.saveSettings());
    this.dom.resetSysPrompt.addEventListener('click', () => {
      this.dom.sysPromptInput.value = DEFAULT_SYSTEM_PROMPT;
    });

    this.dom.tempSlider.addEventListener('input', (e) => {
      this.dom.tempVal.textContent = e.target.value;
    });

    // Mobile sidebar toggle
    this.dom.mobileMenuBtn.addEventListener('click', () => {
      this.dom.sidebar.classList.toggle('open');
    });
  }

  initSettingsUI() {
    this.dom.apiEndpointInput.value = this.settings.endpoint;
    this.dom.modelAliasInput.value = this.settings.model;
    this.dom.tempSlider.value = this.settings.temperature;
    this.dom.tempVal.textContent = this.settings.temperature;
    this.dom.maxTokensInput.value = this.settings.maxTokens;
    this.dom.sysPromptInput.value = this.settings.systemPrompt;
  }

  openSettings() {
    this.initSettingsUI();
    this.dom.settingsModal.style.display = 'flex';
  }

  closeSettings() {
    this.dom.settingsModal.style.display = 'none';
  }

  saveSettings() {
    this.settings.endpoint = this.dom.apiEndpointInput.value.trim() || '/v1';
    this.settings.model = this.dom.modelAliasInput.value.trim() || 'jugurtha-4b';
    this.settings.temperature = parseFloat(this.dom.tempSlider.value);
    this.settings.maxTokens = parseInt(this.dom.maxTokensInput.value, 10) || 1024;
    this.settings.systemPrompt = this.dom.sysPromptInput.value.trim() || DEFAULT_SYSTEM_PROMPT;

    localStorage.setItem('jugurtha_endpoint', this.settings.endpoint);
    localStorage.setItem('jugurtha_model', this.settings.model);
    localStorage.setItem('jugurtha_temp', this.settings.temperature);
    localStorage.setItem('jugurtha_max_tokens', this.settings.maxTokens);
    localStorage.setItem('jugurtha_sys_prompt', this.settings.systemPrompt);

    this.closeSettings();
    this.checkServerHealth();
  }

  async checkServerHealth() {
    const endpoint = this.settings.endpoint.replace(/\/$/, '');
    try {
      const res = await fetch(`${endpoint}/models`, { method: 'GET', signal: AbortSignal.timeout(3500) });
      if (res.ok) {
        this.dom.statusDot.className = 'status-dot online';
        this.dom.statusLabel.textContent = 'vLLM Online (H100)';
      } else {
        throw new Error('Not OK');
      }
    } catch {
      this.dom.statusDot.className = 'status-dot';
      this.dom.statusLabel.textContent = 'Server Offline';
    }
  }

  startNewChat() {
    if (this.isGenerating) this.stopGeneration();
    this.messages = [];
    this.dom.thread.innerHTML = '';
    this.dom.welcomeHero.style.display = 'block';
  }

  async handleUserSend() {
    const text = this.dom.userInput.value.trim();
    if (!text || this.isGenerating) return;

    this.dom.welcomeHero.style.display = 'none';
    this.dom.userInput.value = '';
    this.dom.userInput.style.height = 'auto';

    // Append user message to DOM and state
    this.appendUserMessage(text);
    this.messages.push({ role: 'user', content: text });

    // Prepare assistant message container
    const assistantElements = this.createAssistantMessageContainer();
    this.scrollToBottom();

    // Launch streaming request
    await this.streamAssistantResponse(assistantElements);
  }

  appendUserMessage(text) {
    const row = document.createElement('div');
    row.className = 'message-row user';
    const bubble = document.createElement('div');
    bubble.className = 'message-bubble user';
    this.renderMarkdown(bubble, text);
    row.appendChild(bubble);
    this.dom.thread.appendChild(row);
  }

  createAssistantMessageContainer() {
    const row = document.createElement('div');
    row.className = 'message-row assistant';

    const avatar = document.createElement('div');
    avatar.className = 'assistant-avatar';
    avatar.textContent = 'ⵣ';

    const wrapper = document.createElement('div');
    wrapper.className = 'assistant-content-wrapper';

    // Thinking container
    const thinkingCard = document.createElement('div');
    thinkingCard.className = 'thinking-card';
    thinkingCard.style.display = 'none'; // Hidden until <think> token seen
    thinkingCard.innerHTML = `
      <div class="thinking-header">
        <div class="thinking-title-area">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 4.44-2.04z"></path><path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-4.44-2.04z"></path></svg>
          <span>Asmekti n Jugurtha (Reasoning)</span>
          <span class="thinking-badge" id="think-timer">La yettxemmim...</span>
        </div>
        <svg class="thinking-chevron" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"></polyline></svg>
      </div>
      <div class="thinking-body"></div>
    `;

    thinkingCard.querySelector('.thinking-header').addEventListener('click', () => {
      thinkingCard.classList.toggle('collapsed');
    });

    // Main text bubble
    const bubble = document.createElement('div');
    bubble.className = 'message-bubble assistant';

    // Meta stats bar
    const metaBar = document.createElement('div');
    metaBar.className = 'message-meta-bar';
    metaBar.style.display = 'none';

    wrapper.appendChild(thinkingCard);
    wrapper.appendChild(bubble);
    wrapper.appendChild(metaBar);

    row.appendChild(avatar);
    row.appendChild(wrapper);
    this.dom.thread.appendChild(row);

    return {
      row,
      thinkingCard,
      thinkingBody: thinkingCard.querySelector('.thinking-body'),
      thinkingBadge: thinkingCard.querySelector('#think-timer'),
      bubble,
      metaBar
    };
  }

  async streamAssistantResponse(elements) {
    this.isGenerating = true;
    this.dom.sendBtn.style.display = 'none';
    this.dom.stopBtn.style.display = 'flex';
    this.abortController = new AbortController();

    const startTime = performance.now();
    let firstTokenTime = null;
    let tokenCount = 0;
    let fullResponse = '';
    let isInsideThink = false;
    let thinkContent = '';
    let answerContent = '';

    // Build payload messages including system prompt
    const payloadMessages = [
      { role: 'system', content: this.settings.systemPrompt },
      ...this.messages
    ];

    const endpoint = this.settings.endpoint.replace(/\/$/, '') + '/chat/completions';

    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: this.abortController.signal,
        body: JSON.stringify({
          model: this.settings.model,
          messages: payloadMessages,
          temperature: this.settings.temperature,
          max_tokens: this.settings.maxTokens,
          stream: true,
          extra_body: {
            enable_thinking: this.settings.enableThinking
          }
        })
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP ${response.status}: ${errorText}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop(); // Keep partial line

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed || !trimmed.startsWith('data:')) continue;
          if (trimmed === 'data: [DONE]') break;

          try {
            const data = JSON.parse(trimmed.slice(5).trim());
            const delta = data.choices?.[0]?.delta?.content || '';
            if (!delta) continue;

            if (firstTokenTime === null) firstTokenTime = performance.now();
            tokenCount++;
            fullResponse += delta;

            // Handle <think> tags dynamically
            if (fullResponse.includes('<think>') && !isInsideThink && !fullResponse.includes('</think>')) {
              isInsideThink = true;
              elements.thinkingCard.style.display = 'block';
            }

            if (isInsideThink) {
              if (fullResponse.includes('</think>')) {
                // Thinking finished
                isInsideThink = false;
                const parts = fullResponse.split('</think>');
                thinkContent = parts[0].replace('<think>', '').trim();
                answerContent = parts.slice(1).join('</think>').trim();
                
                this.renderMarkdown(elements.thinkingBody, thinkContent);
                const thinkDuration = ((performance.now() - startTime) / 1000).toFixed(1);
                elements.thinkingBadge.textContent = `Smekti deg ${thinkDuration}s`;
              } else {
                thinkContent = fullResponse.replace('<think>', '');
                elements.thinkingBody.textContent = thinkContent;
              }
            } else {
              if (fullResponse.includes('</think>')) {
                answerContent = fullResponse.split('</think>').slice(1).join('').trim();
              } else {
                answerContent = fullResponse;
              }
              this.renderMarkdown(elements.bubble, answerContent);
            }

            this.scrollToBottom();
          } catch {
            // Ignore partial SSE chunk parse errors
          }
        }
      }

      // Final rendering pass
      if (thinkContent) {
        this.renderMarkdown(elements.thinkingBody, thinkContent);
      }
      this.renderMarkdown(elements.bubble, answerContent || fullResponse);
      this.messages.push({ role: 'assistant', content: fullResponse });

      // Performance stats
      const totalElapsed = (performance.now() - startTime) / 1000;
      const speed = tokenCount > 0 ? (tokenCount / totalElapsed).toFixed(1) : 0;
      elements.metaBar.style.display = 'flex';
      elements.metaBar.innerHTML = `
        <span class="speed-badge">⚡ ${speed} tok/s</span>
        <span>${tokenCount} tokens</span>
        <span>${totalElapsed.toFixed(2)}s</span>
      `;

    } catch (err) {
      if (err.name === 'AbortError') {
        elements.bubble.innerHTML += `<p><em>[Asuffeɣ yettwaḥbes (Generation stopped)]</em></p>`;
      } else {
        elements.bubble.innerHTML = `
          <div style="color: #EF4444; background: rgba(239, 68, 68, 0.1); padding: 12px; border-radius: 8px; border: 1px solid rgba(239, 68, 68, 0.3);">
            <strong>Tuccḍa deg tuqqna (Connection Error):</strong><br>
            ${this.escapeHTML(err.message)}<br><br>
            <small>Verify that vLLM is running on the HPC (<code>./tools/serve_jugurtha.sh status</code>) and that the endpoint URL in Settings is reachable.</small>
          </div>
        `;
      }
    } finally {
      this.isGenerating = false;
      this.dom.sendBtn.style.display = 'flex';
      this.dom.stopBtn.style.display = 'none';
      this.abortController = null;
    }
  }

  stopGeneration() {
    if (this.abortController) {
      this.abortController.abort();
    }
  }

  renderMarkdown(container, text) {
    if (!text) {
      container.innerHTML = '';
      return;
    }

    try {
      // Step 1: Protect code blocks (fenced ```...``` and inline `...`)
      const codeBlocks = [];
      let protectedText = text.replace(/(```[\s\S]*?```|`[^`\n]+`)/g, (match) => {
        const token = `@@@AGBALU_CODE_${codeBlocks.length}@@@`;
        codeBlocks.push(match);
        return token;
      });

      // Step 2: Extract Display Math ($$...$$ and \[...\] and \begin{env}...\end{env})
      const mathBlocks = [];

      // Display math: $$ ... $$
      protectedText = protectedText.replace(/\$\$([\s\S]*?)\$\$/g, (_, math) => {
        const token = `@@@AGBALU_MATH_BLOCK_${mathBlocks.length}@@@`;
        mathBlocks.push({ math: math.trim(), display: true });
        return token;
      });

      // Display math: \[ ... \]
      protectedText = protectedText.replace(/\\\[([\s\S]*?)\\\]/g, (_, math) => {
        const token = `@@@AGBALU_MATH_BLOCK_${mathBlocks.length}@@@`;
        mathBlocks.push({ math: math.trim(), display: true });
        return token;
      });

      // Display math environments: \begin{equation|align|...} ... \end{...}
      protectedText = protectedText.replace(/(\\begin\{(equation|align|aligned|gather|alignat|matrix|pmatrix|bmatrix|vmatrix|cases)\*?\}[\s\S]*?\\end\{\2\*?\})/g, (_, math) => {
        const token = `@@@AGBALU_MATH_BLOCK_${mathBlocks.length}@@@`;
        mathBlocks.push({ math: math.trim(), display: true });
        return token;
      });

      // Step 3: Extract Inline Math (\(...\) and $...$)
      // Inline math: \( ... \)
      protectedText = protectedText.replace(/\\\(([\s\S]*?)\\\)/g, (_, math) => {
        const token = `@@@AGBALU_MATH_INLINE_${mathBlocks.length}@@@`;
        mathBlocks.push({ math: math.trim(), display: false });
        return token;
      });

      // Inline math: $ ... $ (excluding empty or whitespace-only)
      protectedText = protectedText.replace(/(^|[^\\])\$([^\s$](?:[^$\n]*?[^\s$])?)\$/g, (_, prefix, math) => {
        const token = `@@@AGBALU_MATH_INLINE_${mathBlocks.length}@@@`;
        mathBlocks.push({ math: math.trim(), display: false });
        return prefix + token;
      });

      // Step 4: Restore code blocks so Marked can parse them properly
      protectedText = protectedText.replace(/@@@AGBALU_CODE_(\d+)@@@/g, (_, index) => {
        return codeBlocks[parseInt(index, 10)] || '';
      });

      // Step 5: Parse with Marked
      let html = window.marked ? window.marked.parse(protectedText) : protectedText;

      // Clean up standalone paragraph wrapping around block math
      html = html.replace(/<p>\s*(@@@AGBALU_MATH_BLOCK_\d+@@@)\s*<\/p>/g, '$1');

      // Step 6: Render KaTeX for each math placeholder
      html = html.replace(/@@@AGBALU_MATH_(BLOCK|INLINE)_(\d+)@@@/g, (match, type, indexStr) => {
        const item = mathBlocks[parseInt(indexStr, 10)];
        if (!item) return match;

        if (window.katex) {
          try {
            return window.katex.renderToString(item.math, {
              displayMode: item.display,
              throwOnError: false,
              output: 'htmlAndMathml'
            });
          } catch (e) {
            console.warn('KaTeX render error:', e);
            const delim = item.display ? '$$' : '$';
            return `<span class="katex-error" title="${e.message}">${delim}${item.math}${delim}</span>`;
          }
        } else {
          const delim = item.display ? '$$' : '$';
          return `<code>${delim}${item.math}${delim}</code>`;
        }
      });

      container.innerHTML = html;

      // Add copy buttons to code blocks
      container.querySelectorAll('pre').forEach(pre => {
        if (!pre.querySelector('.copy-code-btn')) {
          const btn = document.createElement('button');
          btn.className = 'copy-code-btn';
          btn.textContent = 'Nɣel (Copy)';
          btn.addEventListener('click', () => {
            const code = pre.querySelector('code')?.innerText || pre.innerText;
            navigator.clipboard.writeText(code).then(() => {
              btn.textContent = 'Ittwanɣel! ✓';
              setTimeout(() => { btn.textContent = 'Nɣel (Copy)'; }, 2000);
            });
          });
          pre.appendChild(btn);
        }
      });
    } catch {
      container.textContent = text;
    }
  }

  scrollToBottom() {
    this.dom.container.scrollTop = this.dom.container.scrollHeight;
  }

  escapeHTML(str) {
    const p = document.createElement('p');
    p.appendChild(document.createTextNode(str));
    return p.innerHTML;
  }
}

// Instantiate on DOM load
window.addEventListener('DOMContentLoaded', () => {
  window.jugurthaApp = new JugurthaChatApp();
});
