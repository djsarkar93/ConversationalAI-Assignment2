########################################################################################################################
# Imports
########################################################################################################################
import datetime
import time
import uuid
import healthcare_chatbot_utilities

import streamlit as st



########################################################################################################################
# Functions
########################################################################################################################
# Function to generate an AI response for a given user query
def generate_response(query, ai_response_yield_rate=0.005):
    ai_response = healthcare_chatbot_utilities.generate_answer(user_query=query, top_k=2, threshold=0.9)
    
    # Yield response one character at a time to simulate a typing effect.
    for char in ai_response:
        yield char
        time.sleep(ai_response_yield_rate)


# Function to log the chatbot's conversations
def log_conversation(conversation_data):
    print(conversation_data)


# Function to log the user feedback on the chatbot's response quality
def log_user_feedback(user_feedback):
    print(user_feedback)


# Function to log the user feedback comment on the chatbot's response quality
def log_user_feedback_comment(user_feedback_comment):
    print(user_feedback_comment)


# Function to collect user comments on the chatbot's response.
@st.dialog(
                'Share your suggestions with us here ...', 
                width='small'
          )
def collect_user_comment(feedback_id, message_id):
    # Get user comment
    user_comment = st.text_area('', height=300)
    
    # Log user comment
    if st.button('Submit') and user_comment:
        # Randomly generate a comment id
        feedback_comment_id = str(uuid.uuid4())
        print(f"Initialized Feeback Comment ID: {feedback_comment_id} [ for Feedback ID: {feedback_id} ]")
        
        # Log the user feedback comment
        user_feedback_comment = {
                                    'time': datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f'),
                                    'conversation_id': st.session_state.conversation_id,
                                    'message_id': message_id,
                                    'feedback_id': feedback_id,
                                    'feedback_comment_id': feedback_comment_id,
                                    'feedback_comment': user_comment
                                }
        log_user_feedback_comment(user_feedback_comment)
        print(f'Feedback comment ID: {feedback_comment_id} was logged!')
        
        # Toast a thanks message
        st.toast("Thanks for sharing your thoughts! We'll use them to improve.", icon='🌱')
        time.sleep(3)
        st.rerun()


# Function to process the 'thumbs down' user feedback (on the chatbot's response quality)
def process_thumbs_down_user_feedback(message_id, ai_message_context):
    # Randomly generate a feedback id
    feedback_id = str(uuid.uuid4())
    print(f"Initialized Feedback ID: {feedback_id} [ for Message ID: {message_id} ]")
    
    # Log the user feedback
    user_feedback = {
                        'time': datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f'),
                        'conversation_id': st.session_state.conversation_id,
                        'message_id': message_id,
                        'feedback_id': feedback_id,
                        'positive_user_feedback_flag': False
                    }
    log_user_feedback(user_feedback)
    print(f'Feedback ID: {feedback_id} was logged!')
    
    # Divide AI message context into 2 columns
    col1, col2 = ai_message_context.columns(
                                                [.55, .45], 
                                                gap                = 'small', 
                                                vertical_alignment = 'bottom'
                                           )
    
    # Display 'We're sorry the response wasn't quite right!' in column #1 
    col1.markdown(
                    """<p style="background-color: #3D3C3A;
                                 padding: 7px;
                                 border-radius: 5px;
                                 text-align: center;">
                            😟&nbsp;&nbsp;&nbsp;We're sorry the response wasn't quite right!
                        </p>
                    """,
                    unsafe_allow_html=True
                 )
    
    # Display collect feedback button in column #2
    col2.button(
                    'Help us get better with your feedback', 
                    use_container_width = True,
                    type                = 'primary', 
                    on_click            = collect_user_comment,
                    args                = [feedback_id, message_id]
               )


# Function to process the 'thumbs up' user feedback (on the chatbot's response quality)   
def process_thumbs_up_user_feedback(message_id):
    # Randomly generate a feedback id
    feedback_id = str(uuid.uuid4())
    print(f"Initialized Feedback ID: {feedback_id} [ for Message ID: {message_id} ]")
    
    # Log the user feedback
    user_feedback = {
                        'time': datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f'),
                        'conversation_id': st.session_state.conversation_id,
                        'message_id': message_id,
                        'feedback_id': feedback_id,
                        'positive_user_feedback_flag': True
                    }
    log_user_feedback(user_feedback)
    print(f'Feedback ID: {feedback_id} was logged!')
    
    # Toast a thanks message
    st.toast(
                "Thanks for your feedback! We're glad you enjoyed it.", 
                icon = '🎉'
            )


# Function to request for user feedback on the chatbot's response quality
def request_for_user_feedback_on_ai_response(message_id, ai_message_context):
    # Divide AI message context into 3 columns
    col1, col2, col3 = st.columns(
                                    [.8, .1, .1], 
                                    gap                = 'small', 
                                    vertical_alignment = 'bottom'
                                 )
    
    # Display 'How was the response?' in column #1
    with col1:
        st.markdown(
                        """<p style="background-color: #3D3C3A;
                                    padding: 7px;
                                    border-radius: 5px;
                                    text-align: center;">
                                🤔&nbsp;&nbsp;&nbsp;How was the response?
                            </p>
                        """,
                        unsafe_allow_html = True
                   )
    
    # Display 'thumbs up' button in column #2
    with col2:
        st.button(
                    '👍', 
                    use_container_width = True, 
                    on_click            = process_thumbs_up_user_feedback,
                    args                = [message_id]
                 )
    
    # Display 'thumbs down' button in column #3
    with col3:
        st.button(
                    '👎', 
                    use_container_width = True, 
                    on_click            = process_thumbs_down_user_feedback, 
                    args                = [message_id, ai_message_context]
                 )


# Function to initialize and manage the chatbot
def launch_app():
    # Initialize the chatbot's conversation id & message history list
    if 'messages' not in st.session_state:
        # Randomly generate a conversation id
        st.session_state.conversation_id = str(uuid.uuid4())
        print(f"Initialized Conversation ID: {st.session_state.conversation_id}")
        
        # Assign empty list as message history
        st.session_state.messages = []
    
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
    
    # Accept user query
    if user_query := st.chat_input('Type a message here.'):
        # Randomly generate a message id
        message_id = str(uuid.uuid4())
        print(f"Initialized Message ID: {message_id} [ for Conversation ID: {st.session_state.conversation_id} ]")
        
        # Display user
        with st.chat_message('user'):
            st.markdown(user_query)
        
        # Add user query to the chat history
        st.session_state.messages.append(   
                                            {
                                                'role': 'user', 
                                                'content': user_query
                                            }
                                        )
        
        # Generate chatbot response
        ai_response_stream = generate_response(query = user_query)
        
        # Display chatbot response
        ai_message_context = st.chat_message('assistant')
        ai_response = ai_message_context.write_stream(ai_response_stream)
        
        # Add chatbot response to the chat history
        st.session_state.messages.append(
                                            {
                                                'role': 'assistant', 
                                                'content': ai_response
                                            }
                                        )
        
        # Log user query & chatbot response
        conversation_data = {
                                'time': datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f'),
                                'conversation_id': st.session_state.conversation_id,
                                'message_id': message_id,
                                'user_query': user_query,
                                'chatbot_response': ai_response
                            }
        log_conversation(conversation_data)
        print(f"Message ID: {message_id} was logged!")
        
        # Request for user feedback
        request_for_user_feedback_on_ai_response(message_id, ai_message_context)



########################################################################################################################
# Main
########################################################################################################################
if __name__ == '__main__':
    # Set up the page title, layout, and icon
    st.set_page_config(
                            page_title = 'CAI Asgmt-2 PS-A Group-6', 
                            page_icon  = 'https://upload.wikimedia.org/wikipedia/commons/0/05/Robot_icon.svg', 
                            layout     = 'centered'
                      )
    

    # Display the headers and team information
    st.markdown(
                    """<h3 style="text-align: center;">Conversational AI</h3>
                       <h5 style="text-align: center;">Assignment 2: Problem Statement A (Health Care Chatbot)</h5>
                       <h6 style="text-align: center;">By</h6>
                       <h5 style="text-align: center;">Group 6</h5>
                       <table style="width:50%; margin-left:auto; margin-right:auto; text-align: center;">
                          <tr><th>S/N</th><th>Team Member</th><th>BITS ID</th></tr>
                          <tr><td>1</td><td>Gokul K</td><td>2022AC05398</td></tr>
                          <tr><td>2</td><td>Thirumagal Dhivya S</td><td>2022AC05395</td></tr>
                          <tr><td>3</td><td>Dibyajyoti Sarkar</td><td>2022AA05005</td></tr>
                       </table>
                       <hr/>""", 
                    unsafe_allow_html = True
               )
    
    
    # Display the disclaimer
    st.info(
                'This app is not a real healthcare assistant and may provide inaccurate information or make mistakes; always consult an actual healthcare professional for medical advice.', 
                icon = 'ℹ️'
           )
   

   # Start the chatbot application
    launch_app()