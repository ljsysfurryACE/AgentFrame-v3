/**
 * Smoke test for dsh-memory-director.
 * Uses a mock LLM so no API key is needed.
 */
import { Context } from '@deepseek-ai/cordis'
import { MemoryDirectorService } from '../src/index.js'

/** Minimal mock LLM: returns a canned memory decision. */
const mockLlm = {
  async chat(opts: any) {
    const user = opts.messages?.find((m: any) => m.role === 'user')?.content ?? ''
    if (user.includes('服务器')) {
      return { content: '{"remember":["服务器 IP 是 98.142.241.130","SSH 端口 44123"],"forget":["今天天气不错"],"importance":0.8}' }
    }
    return { content: '{"remember":[],"forget":[],"importance":0.3}' }
  },
}

async function main() {
  const ctx = new Context()
  // Inject mock llm.
  ;(ctx as any).get = (key: string) => (key === 'llm' ? mockLlm : undefined)

  const service = new MemoryDirectorService(ctx, {
    provider: 'mock',
    model: 'mock',
    storePath: '/tmp/dsh-memory-test.json',
    auto: false,
  })

  console.log('[smoke] ctx.memory =', ctx.get('memory') === service ? '✅ registered' : '❌')

  // 1. Direct remember.
  service.remember('用户偏好使用 Python 开发', 0.9)
  console.log('[smoke] remember direct =', service.all().length === 1 ? '✅' : '❌')

  // 2. Dedup: same text again should be skipped.
  service.remember('用户偏好使用 Python 开发', 0.9)
  console.log('[smoke] dedup =', service.all().length === 1 ? '✅' : '❌')

  // 3. LLM decision on a server-info turn.
  const decision = await service.decide('用户说：我的服务器 IP 是 98.142.241.130，SSH 端口 44123。今天天气不错。')
  console.log('[smoke] LLM decision =', JSON.stringify(decision))
  console.log('[smoke] remember 2 items =', decision.remember.length === 2 ? '✅' : '❌')
  console.log('[smoke] forget 1 item =', decision.forget.length === 1 ? '✅' : '❌')

  // 4. Apply decision.
  decision.remember.forEach((item: string) => service.remember(item, decision.importance))
  console.log('[smoke] total memories =', service.all().length === 3 ? '✅ (1 direct + 2 LLM)' : `❌ (${service.all().length})`)

  // 5. Search relevance.
  const results = service.search('服务器 IP 多少？', 3)
  console.log('[smoke] search hits =', results.length >= 1 ? '✅' : '❌')
  console.log('[smoke] search top =', results[0]?.text?.slice(0, 40))

  // 6. Forget.
  const id = service.all()[0]?.id
  if (id) service.forget(id)
  console.log('[smoke] forget =', service.all().length === 2 ? '✅' : '❌')

  console.log('\n✅ smoke test complete')
}

main().catch((e) => {
  console.error('❌ smoke test failed:', e)
  process.exit(1)
})
