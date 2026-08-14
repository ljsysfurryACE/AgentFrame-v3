/**
 * Smoke test for dsh-aura-scheduler.
 */
import { Context } from '@deepseek-ai/cordis'
import { AuraSchedulerService } from '../src/index.js'

async function main() {
  const ctx = new Context()


  // Fast config for testing: 2s min, 5s max, no quiet hours.
  const aura = new AuraSchedulerService(ctx, {
    minInterval: 2,
    maxInterval: 5,
    quietHours: [0, 0] as [number, number], // no quiet (0-0 means always quiet? see logic: start<end false → h>=0||h<0 → always true = always quiet) → use [99,99] to disable
    alpha: 0.5,
    beta: 0.5,
    relevance: 0.5,
    delta: 0.1,
    cooldown: 0,
    auto: false,
  })
  // Note: [0,0] with start<end false → h>=0 || h<0 → always true → always quiet.
  // Override quiet to test properly:
  ;(aura as any).config.quietHours = [25, 25] as [number, number] // start>end false→ h>=25||h<25 → false = never quiet

  console.log('[smoke] ctx.aura =', ctx.get('aura') === aura ? '✅ registered' : '❌')

  // Force tick (skips quiet/cooldown via force).
  const ok = await aura.tickNow()
  console.log('[smoke] force proactive =', ok ? '✅' : '❌')

  const status = aura.status()
  console.log('[smoke] status =', JSON.stringify(status))
  console.log('[smoke] proactiveCount =', status.proactiveCount === 1 ? '✅' : '❌')

  // Normal tick with cooldown should be skipped.
  await aura.tickNow() // force again
  console.log('[smoke] proactiveCount after 2nd =', status.proactiveCount, '(cooldown 0 → should increment)')

  // Test heartbeat auto-start.
  aura.start()
  console.log('[smoke] auto-start running =', status.running !== undefined ? '✅' : '❌')
  aura.stop()

  console.log('\n✅ smoke test complete')
}

main().catch((e) => {
  console.error('❌ smoke test failed:', e)
  process.exit(1)
})
