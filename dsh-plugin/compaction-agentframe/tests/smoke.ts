/**
 * Smoke test: load AgentFrameCompactionEngine in a Cordis context.
 */
import { Context } from '@deepseek-ai/cordis'
import { AgentFrameCompactionEngine } from '../src/index.js'

async function main() {
  const ctx = new Context()

  // Register the AgentFrame compaction engine.
  const engine = new AgentFrameCompactionEngine(ctx, {
    semantic: true,
    retainRatio: 0.2,
    physical: true,
    bytesPerToken: 7776,
    auto: false,
  })

  // Verify it's exposed on ctx.compaction.
  const registered = ctx.get('compaction')
  console.log('[smoke] ctx.compaction =', registered ? '✅ registered' : '❌ missing')

  // Verify class identity.
  console.log('[smoke] engine instanceof AgentFrameCompactionEngine =',
    engine instanceof AgentFrameCompactionEngine ? '✅' : '❌')

  // Verify config.
  console.log('[smoke] config =', JSON.stringify(engine.config))

  // Verify abstract methods are implemented.
  const methods = ['compactIfNeeded', 'compactNow', 'compactRegion']
  for (const m of methods) {
    console.log(`[smoke] ${m} =`, typeof (engine as any)[m] === 'function' ? '✅' : '❌')
  }

  // Verify condense logic with a fake session.
  const fakeSession = {
    log: [
      { seq: 0, type: 'user/message', content: '帮我写一个 TCP 服务器' },
      { seq: 1, type: 'assistant/message', content: '好的，用 Python 的 socket 模块。先创建 socket: `s = socket.socket()`' },
      { seq: 2, type: 'user/message', content: '今天天气不错' },
      { seq: 3, type: 'assistant/message', content: '哈哈是啊。继续：绑定端口 8080，监听连接。' },
      { seq: 4, type: 'user/message', content: '那错误处理怎么做？' },
      { seq: 5, type: 'assistant/message', content: '用 try/except 捕获 ConnectionError，记录日志。这是关键点。' },
      { seq: 6, type: 'user/message', content: '好的明白了' },
      { seq: 7, type: 'assistant/message', content: '嗯嗯有问题随时问～' },
    ],
  }

  const span = (engine as any)._selectSpan(fakeSession)
  console.log('\n[smoke] selected span =', span)

  const summary = (engine as any)._condense(fakeSession, span.start, span.end)
  console.log('[smoke] condensed summary:')
  console.log('----------------------------------------')
  console.log(summary)
  console.log('----------------------------------------')

  // Verify chatter was dropped (no "天气不错" in summary).
  const chatterDropped = !summary.includes('天气不错')
  const codeKept = summary.includes('socket')
  console.log('[smoke] 闲聊被剔除 =', chatterDropped ? '✅' : '❌')
  console.log('[smoke] 关键代码保留 =', codeKept ? '✅' : '❌')

  console.log('\n✅ smoke test complete')
}

main().catch((e) => {
  console.error('❌ smoke test failed:', e)
  process.exit(1)
})
