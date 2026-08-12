import asyncio
from autfin.validation import (
    validate_guardrails, validate_memory_persistence, 
    validate_plan_persistence, validate_eval_suite,
    validate_load_test, validate_durable_execution,
    generate_report
)

async def test():
    results = []
    
    print('Testing guardrails...')
    result = await validate_guardrails()
    status = "PASS" if result.passed else "FAIL"
    print(f'  Guardrails: {status} - {result.message}')
    results.append(result)
    
    print('Testing memory persistence...')
    result = await validate_memory_persistence()
    status = "PASS" if result.passed else "FAIL"
    print(f'  Memory: {status} - {result.message[:100]}')
    results.append(result)
    
    print('Testing plan persistence...')
    result = await validate_plan_persistence()
    status = "PASS" if result.passed else "FAIL"
    print(f'  Plan: {status} - {result.message[:100]}')
    results.append(result)
    
    print('Testing eval suite...')
    result = await validate_eval_suite()
    status = "PASS" if result.passed else "FAIL"
    print(f'  Eval: {status} - {result.message[:100]}')
    results.append(result)
    
    print('Testing load test...')
    result = await validate_load_test()
    status = "PASS" if result.passed else "FAIL"
    print(f'  Load: {status} - {result.message[:100]}')
    results.append(result)
    
    print('Testing durable execution...')
    result = await validate_durable_execution()
    status = "PASS" if result.passed else "FAIL"
    print(f'  Durable: {status} - {result.message[:100]}')
    results.append(result)
    
    from autfin.validation import generate_report
    print(generate_report(results))

asyncio.run(test())