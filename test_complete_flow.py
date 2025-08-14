#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, 'src')
import asyncio

# Set environment variables
os.environ['UTILITIES_CONFIG'] = 'config.local.ini'
os.environ['USE_MOCK_SEARCH'] = 'true'
os.environ['USE_LOCAL_AZURE'] = 'true'
os.environ['OPENAI_API_VERSION'] = '2024-02-15-preview'

async def test_complete_flow():
    print('🔍 Testing Complete LangGraph Flow with State Management Fix')
    print('=' * 60)
    
    try:
        # Initialize resources first
        from infra.resource_manager import initialize_resources, get_resources
        from src.infra.settings import get_settings
        
        print('Initializing resources...')
        settings = get_settings()
        resources = initialize_resources(settings)
        print('✅ Resources initialized')
        
        # Now test the complete flow
        from controllers.graph_integration import handle_turn
        
        user_input = 'tell me about ETU'
        print(f'Testing complete flow with: "{user_input}"')
        
        response_count = 0
        final_result = None
        has_infinite_loop = False
        
        async for update in handle_turn(
            user_input=user_input,
            resources=resources,
            chat_history=[],
            thread_id='test_complete_flow',
            user_context={'user_id': 'test_user', 'session_metadata': {'cloud_profile': 'local'}}
        ):
            response_count += 1
            update_type = update.get('type', 'unknown')
            message = update.get('message', '')
            
            print(f'Update {response_count}: {update_type} - {message[:100]}...')
            
            # Check for completion
            if update_type == 'complete':
                final_result = update
                print('✅ Got successful completion!')
                break
            elif update_type == 'error':
                print(f'❌ Got error: {message}')
                if 'GRAPH_RECURSION_LIMIT' in message:
                    has_infinite_loop = True
                break
            
            # Detect infinite loop pattern
            if response_count > 15:
                print('⚠️  Detected potential infinite loop, breaking...')
                has_infinite_loop = True
                break
        
        if has_infinite_loop:
            print('\n❌ INFINITE LOOP STILL EXISTS')
            print('The state management fix did not resolve the recursion issue')
            return False
        elif final_result:
            answer = final_result.get('result', {}).get('answer', 'No answer')
            print(f'\n✅ SUCCESS: Got final answer: "{answer[:200]}..."')
            print(f'Total updates: {response_count}')
            return True
        else:
            print('\n⚠️  No final result received')
            return False
            
    except Exception as e:
        print(f'\n❌ Test failed with exception: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_complete_flow())
    
    if success:
        print('\n🎉 COMPLETE INFINITE LOOP ISSUE RESOLVED!')
        print('✅ State management fix successful')
        print('✅ ETU query processes without infinite loops') 
        print('✅ Original user issue completely solved')
    else:
        print('\n⚠️  Issue may still exist - further investigation needed')