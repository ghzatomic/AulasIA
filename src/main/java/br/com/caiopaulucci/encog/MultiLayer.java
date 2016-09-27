package br.com.caiopaulucci.encog;

import java.util.Arrays;

import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.back.Backpropagation;

public class MultiLayer {

	// Identificar Laranja, Tangerina, Maca, Morango

	// Mais acidez com mais cor laranja - Laranja
	// Mais fosforo com mais cor laranja - Tangerina
	// Mais acidez com mais cor laranja - Maca
	// Mais fosforo com mais cor laranja - Morango

	// 1 - Acidez
	// 2 - Fosforo
	// 3 - Coloracao (Proximo de 0 laranja, proximo de 1 Vermelho)

	public static void main(String[] args) {
		
		BasicNetwork network = new BasicNetwork();

		network.addLayer(new BasicLayer(null, true, 3));
		network.addLayer(new BasicLayer(new ActivationSigmoid(), false, 2));

		network.getStructure().finalizeStructure();
		// Finaliza a constru��o da rede, inicializando os pesos (w)
		network.reset();

		// Dados de treinamento
		MLDataSet trainingSet = new BasicMLDataSet(getValoresTreinamento(), getValoresSaidaTreinamento());
		
		// Tipo de treinamento
		//ResilientPropagation train = new ResilientPropagation(network, trainingSet);
		Backpropagation train = new Backpropagation(network, trainingSet);

		double minimoErro = 0.01;
		int maximoIteracoes = 100;
		int iteracao = 1;
		
		//Loop de treinamento
		do {
			//Efetua a iteracao
			train.iteration();

			System.out.println( "Iteracao #" + iteracao + " Erro:" + train.getError() );
			iteracao++;
		} while ( train.getError() > minimoErro && iteracao < maximoIteracoes );

		//Conclui o treinamento
		train.finishTraining();
		
		double[] valoresMacaTeste1 = new double[]{ 0.60, 0.4, 1.0 };
		double[] valoresLaranjaTeste1 = new double[]{ 0.58, 0.22,  0.0 };
		double[] valoresTangerinaTeste1 = new double[]{0.10, 0.86, 0.0};
		double[] valoresMorangoTeste1 = new double[]{0.0,0.90,1.0};
		
		MLDataSet amostraTeste = criarTeste(valoresLaranjaTeste1);
		
		MLData pesosComputados = network.compute( amostraTeste.get(0).getInput() );
		
		//System.out.println(pegarSaida(pesosComputados));
		
		testarRede(network, getValoresTreinamento(),getValoresSaidaTreinamento());

	}
	
	//Cria o conjunto de testes
	public static MLDataSet criarTeste( double[ ] input ) {
		final MLDataSet amostraTeste = new BasicMLDataSet();
		MLData ml = new BasicMLData( input );
		amostraTeste.add(ml);
		return amostraTeste;
	}

	public static void testarRede( BasicNetwork network ,double[ ][ ] input ,double[ ][ ] ok) {
		int i = 0;
		for (double[] testado : input) {
			String valrEsperado = pegarSaidaDouble(ok[i]);
			
			MLData pesosComputados = network.compute(criarTeste(testado).get(0).getInput());
			System.out.println(i+" - Esperado : "+valrEsperado+" Recebido : "+pegarSaida(pesosComputados));
			i++;
		}
	}
	
	
	
	public static String pegarSaida(MLData saida) {
		return pegarSaidaDouble(ajustarSaida(saida.getData()));
	}
	
	// Ajusta a saida
	public static String pegarSaidaDouble(double[] saida) {
		double[] resultado = saida;
		if (resultado[0] == 0.0 && resultado[1] == 0.0){
			return "LARANJA";
		}else if (resultado[0] == 0.0 && resultado[1] == 1.0){
			return "TANGERINA";
		}else if (resultado[0] == 1.0 && resultado[1] == 0.0){
			return "MACA";
		}else if (resultado[0] == 1.0 && resultado[1] == 1.0){
			return "MORANGO";
		}
		return "NAO SEI";
		
	}
	
	
	public static double[ ] ajustarSaida( double[ ] resultado ) {
		double[ ] retorno  = new double[resultado.length];
		for ( int i = 0; i < resultado.length; i++ ) {
			double valor = resultado[ i ];
			if ( valor >= 0.5 ) { // Threshold
				retorno[i] = 1;
			} else {
				retorno[i] = 0;
			}
		}
		return retorno;
	}

	public static double[][] getValoresSaidaTreinamento(){
		return new double[][]{ 
				{ 0, 0 }, // Laranja 0
				{ 0, 0 }, // Laranja 1
				{ 0, 0 }, // Laranja 2 
				{ 0, 0 }, // Laranja 3
				{ 0, 0 }, // Laranja 4
				{ 0, 1 }, // Tangerina 5
				{ 0, 1 }, // Tangerina 6
				{ 0, 1 }, // Tangerina 7
				{ 0, 1 }, // Tangerina 8
				{ 0, 1 }, // Tangerina 9
				{ 0, 1 }, // Tangerina 10
				{ 1, 0 }, // Maca 11
				{ 1, 0 }, // Maca 12
				{ 1, 0 }, // Maca 13 
				{ 1, 0 }, // Maca 14
				{ 1, 1 }, // Morango 15
				{ 1, 1 }, // Morango 16 
				{ 1, 1 }, // Morango 17
				{ 1, 1 }  // Morango 18
			};
	}
	
	public static double[][] getValoresTreinamento(){
		return new double[][]{ 
			{ 0.98, 0.1,  0.0 }, // Laranja
			{ 0.70, 0.12, 0.0 }, // Laranja
			{ 0.60, 0.40, 0.0 }, // Laranja
			{ 0.68, 0.2,  0.0 }, // Laranja
			{ 0.75, 0.49, 0.0 }, // Laranja
			{ 0.13, 0.97, 0.0 }, // Tangerina
			{ 0.32, 0.78, 0.0 }, // Tangerina
			{ 0.24, 0.60, 0.0 }, // Tangerina
			{ 0.20, 0.83, 0.0 }, // Tangerina
			{ 0.39, 0.70, 0.0 }, // Tangerina
			{ 0.19, 0.65, 0.0 }, // Tangerina
			{ 0.95, 0.11, 1.0 }, // Maca
			{ 0.65, 0.3, 1.0 },  // Maca
			{ 0.73, 0.1, 1.0 },  // Maca
			{ 0.90, 0.19, 1.0 }, // Maca
			{ 0.57, 0.78, 1.0 }, // Morango
			{ 0.25, 0.84, 1.0 }, // Morango
			{ 0.33, 0.54, 1.0 }, // Morango
			{ 0.48, 0.83, 1.0 }  // Morango
		};
	}
	
}
