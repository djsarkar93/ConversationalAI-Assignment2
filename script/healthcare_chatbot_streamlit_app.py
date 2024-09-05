########################################################################################################################
# Imports
########################################################################################################################
import streamlit as st



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