

import json # use this for serialising dictionaries and lists used in our message https://realpython.com/python-json/
import time
from statemachine import StateMachine # Use the base state machine to deal with some nice complexities


class ClientARQFSM(StateMachine):
    def __init__(self):
        self._state_machine = {'init': {'init': {'nxt_state': 'Wait_Hello_Ack', 'action': self._init},
                                        'Error':{'nxt_state':'Fault_State','action':self.log_error_msg}},
                               'Wait_Data_A': {'Data_Ready': {'nxt_state': 'Wait_Ack_0', 'action': self._send_msg_0},
                                               'Error':{'nxt_state':'Fault_State','action':self.log_error_msg},
                                               'Recv_Ack_1': {'nxt_state': 'Wait_Data_A', 'action': self._ackn_log},
                                               'Recv_Ack_0': {'nxt_state': 'Wait_Data_A', 'action': self._ackn_log},
                                               'Recv_Close':{'nxt_state':'Closed','action':self._recv_close_log},
                                               'Close':{'nxt_state':"Closed",'action':self._close}},
                               'Wait_Ack_0': {'Corrupt': {'nxt_state': 'Wait_Ack_0', 'action': self._resend_msg_0},
                                              'Recv_Ack_1': {'nxt_state': 'Wait_Ack_0', 'action': self._resend_msg_0},
                                              'Recv_Ack_0': {'nxt_state': 'Wait_Data_B', 'action': self._rcv_ack_0},
                                              'Error':{'nxt_state':'Fault_State','action':self.log_error_msg},
                                              'Timeout':{'nxt_state':'Wait_Ack_0','action':self._resend_msg_0},
                                              'Recv_Close':{'nxt_state':'Closed','action':self._recv_close_log},
                                              'Close':{'nxt_state':"Closed",'action':self._close}},
                               'Wait_Data_B': {'Data_Ready': {'nxt_state': 'Wait_Ack_1', 'action': self._send_msg_1},
                                               'Error':{'nxt_state':'Fault_State','action':self.log_error_msg},
                                               'Recv_Ack_1': {'nxt_state': 'Wait_Data_B', 'action': self._ackn_log},
                                               'Recv_Ack_0': {'nxt_state': 'Wait_Data_B', 'action': self._ackn_log},
                                               'Recv_Close':{'nxt_state':'Closed','action':self._recv_close_log},
                                               'Close':{'nxt_state':"Closed",'action':self._close}},
                               'Wait_Ack_1': {'Corrupt': {'nxt_state': 'Wait_Ack_1', 'action': self._resend_msg_1},
                                              'Recv_Ack_0': {'nxt_state': 'Wait_Ack_1', 'action': self._resend_msg_1},
                                              'Recv_Ack_1': {'nxt_state': 'Wait_Data_A', 'action': self._rcv_ack_1},
                                              'Error':{'nxt_state':'Fault_State','action':self.log_error_msg},
                                              'Timeout':{'nxt_state':'Wait_Ack_1','action':self._resend_msg_1},
                                              'Recv_Close':{'nxt_state':'Closed','action':self._recv_close_log},
                                              'Close':{'nxt_state':"Closed",'action':self._close}},
 
                               'Fault_State':{},
                               'Closed': {},
                               'Wait_Hello_Ack': {'Recv_Ack_1': {'nxt_state': 'Wait_Hello_Ack','action': self._wrong_ack},
                                                  'Recv_Ack_0': {'nxt_state': 'Wait_Data_A', 'action': self._rcv_ack_0},
                                                  'Timeout': {'nxt_state': 'Wait_Hello_Ack', 'action': '_send_hello'},
                                                  'Recv_Close': {'nxt_state': 'Closed', 'action': self._recv_close_log},
                                                  'Close': {'nxt_state': "Closed", 'action': self._close},
                                                   'Error': {'nxt_state': 'Fault_State', 'action': self.log_error_msg},
                                                   'Corrupt': {'nxt_state': 'Wait_Hello_Ack', 'action': self.log_corrupt_msg}}

                               }
        self._current_state = 'init'
        self._last_msg = None

    def initialise(self, context):
        if 'init' not in self._state_machine.keys():
            raise Exception("No init state exists in FSM")
        self._current_state = 'init'
        self._last_msg = None
        state, output = self.event_handler('init', context)
        print(f'CLIENT: Initialised FSM. Now in State: {state} with output: {output}')

    def event_handler(self, event, context):
        # Check that the current state exists in the state list
        if self._current_state not in self._state_machine.keys():
            #“CLIENT: Unknown event xxx in state yyy”
            print(f'CLIENT: Unknown event {event} in {self._current_state}')
            # raise Exception(f'current state not in state list {self._current_state}')

        # Record the nxt_state for the transition
        # print(self._current_state,"-----",event,"----",self._last_msg)
        nxt_state = self._state_machine[self._current_state][event]['nxt_state']
        # reteive the action for the transition check if it s alegit function
        # if so execute
        output = None
        action = self._state_machine[self._current_state][event]['action']
        if action is not None:
            output = action(context)

        # set the current state and return what the current state now is
        self._current_state = nxt_state
        return self._current_state, output

    def _init(self, context):
        print(f'CLIENT: init Client Arq')

    def _send_msg_0(self, context):
        msg = {'seq': 0, 'type': 'data', 'data':context}
        self._last_msg = msg
        print(f'CLIENT: Sending message 0: {msg}')
        return msg

    def _resend_msg_0(self, context):
        print(f'CLIENT: Resending message 0: {self._last_msg}')
        return self._last_msg

    def _rcv_ack_0(self, context):
        print(f'CLIENT: Recieved ACK 0: {context}')

    def _send_msg_1(self, context):
        msg = {'seq': 1, 'type': 'data', 'data': context}
        self._last_msg = msg
        print(f'CLIENT: Sending message 1: {msg}')
        return msg

    def _resend_msg_1(self, context):
        print(f'CLIENT: Resending message 1: {self._last_msg}')
        return self._last_msg

    def _rcv_ack_1(self, context):
        print(f'CLIENT: Recieved ACK 1: {context}')

    def log_error_msg(self,context):
        print(f'CLIENT: error event in {self._current_state} moving to Fault_State with context: {context}')

    def log_corrupt_msg(self,context):
        x= 0 if self._current_state.split("_")[-1] == "0" else 1
        print(f'CLIENT: Corrupt Message waiting for ACK {x}. MSG: {context}')

    def _wrong_ack(self,context):
        x = 0 if self._current_state.split("_")[-1] == "0" else 1
        print(f'CLIENT: Wrong Ack received Waiting for {x}, received {context["seq"]}. MSG:{context}')

    def _ackn_log(self,context):
        print(f'CLIENT: ack received waiting for data. MSG: {context}')

    def _recv_close_log(self,context):
        print(f'CLIENT : Close msg received in state {self._current_state}. MSG: {context}')

    def _close(self,context):
        close_msg = {'seq':0,'type':'close','data':None}
        self._last_msg = close_msg
        print(f'CLIENT: Closing in state {self._current_state}.sending MSG: {context}')
        return msg

