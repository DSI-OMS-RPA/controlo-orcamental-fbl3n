# FBL3N –  CONTROLO ORÇAMENTAL

Este RPA tem como objetivo extrair as partidas em aberto da transação FBL3N no SAP, para posterior análise e controlo orçamental.

## Requisitos

- Acesso ao SAP com permissões para executar a transação FBL3N.
- Python 3.x instalado.
- Bibliotecas Python necessárias (ver `requirements.txt`).
- Acesso à base de dados especificada no ficheiro `config.ini`.
- SAP GUI instalado e configurado.
## Configuração

1. Clone o repositório para a sua máquina local.
2. Instale as bibliotecas Python necessárias:
   ```
   pip install -r requirements.txt
   ```
3. Configure o ficheiro `config.ini` com as informações da sua base de dados e SAP.
4. Certifique-se de que o SAP GUI está instalado e configurado corretamente.

## Execução
Para executar o RPA, utilize o seguinte comando:
```
python main.py
```

## Notas
- Certifique-se de que o SAP GUI está aberto e que a sessão está ativa antes de executar o RPA.
- Verifique os logs gerados para monitorizar a execução e identificar possíveis erros.
- Ajuste os parâmetros no ficheiro `config.ini` conforme necessário para o seu ambiente
- Este RPA foi desenvolvido para fins específicos e pode necessitar de adaptações para outros contextos ou transações SAP.

## Licença
Este projeto está licenciado sob a Licença MIT. Veja o ficheiro LICENSE para mais detalhes

## Suporte
Para suporte ou dúvidas, por favor contacte o autor do projeto ou equipa RPA.

# Autor
- Nome: Joselito Coutinho
- Email: joselito.coutinho@cvt.cv
