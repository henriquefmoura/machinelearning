"""
Calculadora Simples
Realiza operações matemáticas básicas: adição, subtração, multiplicação e divisão
"""

def adicionar(a, b):
    """Retorna a soma de dois números"""
    return a + b

def subtrair(a, b):
    """Retorna a diferença entre dois números"""
    return a - b

def multiplicar(a, b):
    """Retorna o produto de dois números"""
    return a * b

def dividir(a, b):
    """Retorna a divisão de dois números"""
    if b == 0:
        return "Erro: divisão por zero!"
    return a / b

def exibir_menu():
    """Exibe o menu de opções"""
    print("\n" + "="*40)
    print("         CALCULADORA SIMPLES")
    print("="*40)
    print("1. Adição (+)")
    print("2. Subtração (-)")
    print("3. Multiplicação (*)")
    print("4. Divisão (/)")
    print("5. Sair")
    print("="*40)

def obter_numeros():
    """Obtém dois números do usuário"""
    try:
        num1 = float(input("Digite o primeiro número: "))
        num2 = float(input("Digite o segundo número: "))
        return num1, num2
    except ValueError:
        print("Erro: por favor, digite números válidos!")
        return None, None

def calcular(operacao, num1, num2):
    """Realiza a operação solicitada"""
    operacoes = {
        '1': ('Adição', adicionar),
        '2': ('Subtração', subtrair),
        '3': ('Multiplicação', multiplicar),
        '4': ('Divisão', dividir)
    }
    
    if operacao in operacoes:
        nome_op, funcao = operacoes[operacao]
        resultado = funcao(num1, num2)
        print(f"\n{nome_op}: {num1} e {num2}")
        print(f"Resultado: {resultado}")
        return resultado
    else:
        print("Operação inválida!")
        return None

def main():
    """Função principal - loop da calculadora"""
    print("\nBem-vindo à Calculadora Simples!")
    
    while True:
        exibir_menu()
        opcao = input("Escolha uma opção (1-5): ").strip()
        
        if opcao == '5':
            print("\nObrigado por usar a calculadora. Até logo!")
            break
        
        if opcao in ['1', '2', '3', '4']:
            num1, num2 = obter_numeros()
            if num1 is not None and num2 is not None:
                calcular(opcao, num1, num2)
        else:
            print("Opção inválida! Digite 1, 2, 3, 4 ou 5.")

if __name__ == "__main__":
    main()
