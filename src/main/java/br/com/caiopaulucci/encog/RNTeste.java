package br.com.caiopaulucci.encog;

import java.util.Arrays;

import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.Propagation;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;

public class RNTeste{
	//Cria o conjunto de treinamento
	public static MLDataSet criarConjuntoTreinamento( double[ ][ ] input, double[ ][ ] ideal ) {
		return new BasicMLDataSet( input, ideal );
	}

	//Obtem o tipo de propagacao do treinamento
	public static Propagation criarPropagacao( BasicNetwork redeNeural, MLDataSet conjuntoTreinamento ) {
		return new ResilientPropagation( redeNeural, conjuntoTreinamento );
	}

	//Efetua o treinamento
	public static void treinar( Propagation treinamento ) {
		double minimoErro = 0.01;
		int maximoIteracoes = 100;
		int iteracao = 1;

		//Loop de treinamento
		do {
			//Efetua a iteracao
			treinamento.iteration();

			System.out.println( "Iteracao #" + iteracao + " Erro:" + treinamento.getError() );
			iteracao++;
		} while ( treinamento.getError() > minimoErro && iteracao < maximoIteracoes );

		//Conclui o treinamento
		treinamento.finishTraining();
	}

	private static final double[ ][ ] INPUT = {
			{ 99.74785861, 0.252141395, 79.51192437, 20.48807563, 82.0554725, 17.9445275, 88.0797078, 11.9202922 },
			{ 99.97517306, 0.024826937, 92.0660409, 7.933959097, 97.2311492, 2.7688508, 88.0797078, 11.9202922 },
			{ 73.61589219, 26.38410781, 0.642123532, 99.35787647, 17.7699644, 82.2300356, 11.9202922, 88.0797078 },
			{ 92.10064578, 7.899354218, 46.45089699, 53.54910301, 35.1328759, 64.8671241, 11.9202922, 88.0797078 },
			{ 24.71418313, 75.28581687, 6.825159141, 93.17484086, 12.1054661, 87.8945339, 11.9202922, 88.0797078 },
			{ 76.96653628, 23.03346372, 17.96718882, 82.03281118, 51.4013308, 48.5986692, 11.9202922, 88.0797078 },
			{ 10.57088545, 89.42911455, 0.081726617, 99.91827338, 8.0572927, 91.9427073, 11.9202922, 88.0797078 },
			{ 98.31870068, 1.681299316, 66.04740474, 33.95259526, 48.7068326, 51.2931674, 11.9202922, 88.0797078 },
			{ 62.21377786, 37.78622214, 14.10917927, 85.89082073, 19.4502489, 80.5497511, 11.9202922, 88.0797078 },
			{ 26.74367798, 73.25632202, 9.863102261, 90.13689774, 2.5985171, 97.4014829, 11.9202922, 88.0797078 },
			{ 94.90790211, 5.092097894, 42.32055154, 57.67944846, 26.427101, 73.572899, 11.9202922, 88.0797078 },
			{ 80.76283298, 19.23716702, 0.959312341, 99.04068766, 25.6295804, 74.3704196, 11.9202922, 88.0797078 },
			{ 8.034503155, 91.96549684, 0.081457561, 99.91854244, 1.053243, 98.946757, 11.9202922, 88.0797078 },
			{ 99.16444451, 0.835555493, 72.17355257, 27.82644743, 80.6174951, 19.3825049, 88.0797078, 11.9202922 } 
	};

	private static final double IDEAL[][] = { { 0.0, 1.0 }, //No (0)
	        { 0.0, 1.0 }, //No (0)
	        { 1.0, 0.0 }, //Yes (1)
	        { 1.0, 0.0 }, //Yes (1)
	        { 1.0, 0.0 }, //Yes (1)
	        { 0.0, 1.0 }, //No (0)
	        { 1.0, 0.0 }, //Yes (1)
	        { 0.0, 1.0 }, //No (0)
	        { 1.0, 0.0 }, //Yes (1)
	        { 1.0, 0.0 }, //Yes (1)
	        { 1.0, 0.0 }, //Yes (1)
	        { 1.0, 0.0 }, //Yes (1)
	        { 1.0, 0.0 }, //Yes (1)
	        { 0.0, 1.0 } //No (0)
	};

	//Cria o conjunto de testes
	public static MLDataSet criarConjuntoTeste( double[ ][ ] input ) {
		MLDataSet amostraTeste = new BasicMLDataSet();
		Arrays.stream( input ).forEach( data -> {
			MLData ml = new BasicMLData( data );
			amostraTeste.add( ml );
		} );
		return amostraTeste;
	}

	//Efetua o teste
	public static void testar( BasicNetwork redeNeural, MLDataSet conjuntoTeste ) {
		int indice = 1;

		for ( MLDataPair teste : conjuntoTeste ) {
			MLData saida = ajustarSaida( redeNeural.compute( teste.getInput() ) );
			System.out.println( indice + "\tatual=" + saida + "\tideal=" + new BasicMLData( IDEAL[ indice - 1 ] ) );
			indice++;
		}
	}

	//Ajusta a saida
	public static MLData ajustarSaida( MLData saida ) {
		double[ ] resultado = saida.getData();
		for ( int i = 0; i < resultado.length; i++ ) {
			double valor = resultado[ i ];
			if ( valor >= 0.5 ) {
				resultado[ i ] = 1;
			} else {
				resultado[ i ] = 0;
			}
		}
		saida.setData( resultado );
		return saida;
	}

	//Método de criação da rede neural
	public static BasicNetwork criarRedeNeural() {
		//Cria a rede neural
		BasicNetwork redeNeural = new BasicNetwork();
		//Adiciona a camada de entrada com 8 neuronios
		redeNeural.addLayer( new BasicLayer( null, true, 8 ) );
		//Adiciona a camada de saída com 2 neuronios e a função de ativação sigmoid
		redeNeural.addLayer( new BasicLayer( new ActivationSigmoid(), false, 2 ) );
		//Cria a estrutura da rede
		redeNeural.getStructure().finalizeStructure();
		//Finaliza a construção da rede, inicializando os pesos (w) com valores aleatórioes
		//e menores que 0
		redeNeural.reset();
		return redeNeural;
	}

	public static void main( String[ ] args ) {
		BasicNetwork redeNeural = criarRedeNeural();
		MLDataSet amostraTreinamento = criarConjuntoTreinamento( INPUT, IDEAL );
		Propagation treinamento = criarPropagacao( redeNeural, amostraTreinamento );
		treinar( treinamento );

		MLDataSet amostraTeste = criarConjuntoTeste( INPUT );
		testar( redeNeural, amostraTeste );
	}
}