#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')
import asyncio

async def test_query_loss():
    print('Testing Where Query Gets Lost in Pipeline')
    print('=' * 50)
    
    from agent.nodes.summarize import summarize_node
    
    test_state = {'original_query': 'tell me about ETU'}
    config = {'configurable': {'thread_id': 'test'}}
    
    print('Input to summarize:', test_state['original_query'])
    
    try:
        result = await summarize_node(test_state, config)
        normalized = result.get('normalized_query', 'MISSING')
        print('Output from summarize:', normalized)
        
        if not normalized or normalized == 'MISSING':
            print('PROBLEM: Summarize node not returning normalized_query')
            return False
        
        from agent.nodes.intent import intent_node
        
        state_for_intent = {**test_state, **result}
        print('Input to intent:', state_for_intent.get('normalized_query', 'MISSING'))
        
        intent_result = await intent_node(state_for_intent, config)
        print('Output from intent: intent=', intent_result.get('intent', 'MISSING'))
        
        final_state = {**state_for_intent, **intent_result}
        final_query = final_state.get('normalized_query', 'MISSING')
        
        print('Final state normalized_query:', final_query)
        
        if final_query and final_query != 'MISSING' and len(final_query) > 5:
            print('SUCCESS: Query preserved through pipeline')
            return True
        else:
            print('PROBLEM: Query lost somewhere in pipeline')
            return False
            
    except Exception as e:
        print('Pipeline test failed:', e)
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_query_loss())
    
    if success:
        print('\nIndividual nodes working - issue may be in LangGraph state management')
    else:
        print('\nFound the problem in individual node execution')