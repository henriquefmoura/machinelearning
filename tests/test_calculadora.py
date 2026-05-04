import math
import unittest

from calculadora import (
    adicionar,
    calcular,
    dividir,
    multiplicar,
    subtrair,
)


class CalculadoraTests(unittest.TestCase):
    def test_operacoes_basicas(self):
        self.assertEqual(adicionar(2, 3), 5)
        self.assertEqual(subtrair(10, 3), 7)
        self.assertEqual(multiplicar(4, 2.5), 10)

    def test_divisao_valida(self):
        self.assertTrue(math.isclose(dividir(10, 4), 2.5))

    def test_divisao_por_zero(self):
        self.assertEqual(dividir(10, 0), "Erro: divisão por zero!")

    def test_calcular_operacao_invalida(self):
        self.assertIsNone(calcular("9", 1, 2))

    def test_calcular_divisao(self):
        self.assertTrue(math.isclose(calcular("4", 8, 2), 4.0))


if __name__ == "__main__":
    unittest.main()
