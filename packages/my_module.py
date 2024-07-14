def greet(name):
    """
    print a greeting message

    Parameter
    ----------
    name: {str} the name of the person
    """
    print(f"Hello, {name.title()}!")


def pet(name, animal='cat'):
    """
    display info about a pet

    Parameters:
    -----------
    name: {str} name of the animal.
    animal: {str} type of the animal The default value is cat
    
    """
    print(f'this is a {animal}. Its name is {name}')


def square(x):
    """
    get the square of x
    
    Parameter
    ---------
    x {numeric} a number
    """
    return x**2
    