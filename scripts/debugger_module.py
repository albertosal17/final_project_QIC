import os

def checkpoint(message, debug = True, verbose=0, message1=None, message2=None, **kwargs):
    """
    @brief Prints a checkpoint message and optionally the values of variables.
    
    @param message A string containing the checkpoint message.
    @param debug A boolean flag to enable/disable the checkpoint message.
    @param verbose An integer flag to enable/disable the verbose message. It take values in {0, 1, 2}.
    @param message1 A string containing the message for verbosity equal to 1.
    @param message2 A string containing the message for verbosity equal to 2.
    @param kwargs Key-value pairs representing the variable names and their values to be printed.
    
    @return None. Prints the message and variable values to the console.

    @note: arguments for printing the eventually variables should be passed with the following structure:
            Considering for example the case you want to print two variables var1 and var2
            checkoint("the message", debug=True, var1=var1, var2=var2)

    """
    if debug :
        print(f"[CHECKPOINT] {message}")
        if kwargs:
            for var_name, value in kwargs.items():
                print(f"  {var_name}: {value}")

        if (verbose==1 or verbose==2) and message1 is not None:
            print(f"  {message1}")
        if verbose==2 and message2 is not None:
            print(f"  {message2}")
        else : 
            pass # do nothing
        
        
    
def error(error_message):
    """
    @brief Prints an error message and 
    
    @param message A string containing the checkpoint message.
    @param kwargs Key-value pairs representing the variable names and their values to be printed.
    
    @return None. Prints the message and variable values to the console.
    """
    print(f"[ERROR] {error_message}")
    print("Stopping execution.")  
    os._exit(0)

    

def warning(warning_message, debug=True):
    """
    @brief Prints an error message and 
    
    @param message A string containing the checkpoint message.
    @param kwargs Key-value pairs representing the variable names and their values to be printed.
    
    @return None. Prints the message and variable values to the console.
    """
    if debug:
        print(f"[WARNING] {warning_message}")
    



