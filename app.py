 

 

import streamlit as st 

import pandas as pd 

import joblib 

 

 

st.set_page_config( 

    page_title="Predicción de Demanda", 

    layout="wide" 

) 

 

 

st.title( 

    "Sistema Predictivo de Demanda Empresarial" 

) 

 

 

st.write( 

    """ 

Esta aplicación utiliza un modelo de Machine Learning 

para estimar la demanda del día siguiente. 

""" 

) 

 

 

@st.cache_resource 

def cargar_modelo(): 

 

    return joblib.load( 

        "modelo_demanda_produccion.joblib" 

    ) 

 

 

modelo = cargar_modelo() 

 

 

st.subheader( 

    "Cargar nuevos datos" 

) 

 

 

archivo = st.file_uploader( 

 

    "Seleccione un archivo CSV", 

 

    type=["csv"] 

 

) 

 

 

if archivo is not None: 

 

    nuevos_datos = pd.read_csv( 

        archivo 

    ) 

 

 

    st.write( 

        "Datos recibidos:" 

    ) 

 

 

    st.dataframe( 

        nuevos_datos.head() 

    ) 

 

 

    if st.button( 

        "Generar predicciones" 

    ): 

 

        try: 

 

            predicciones = ( 

                modelo 

                .predict( 

                    nuevos_datos 

                ) 

            ) 

 

 

            resultado = ( 

                nuevos_datos.copy() 

            ) 

 

 

            resultado[ 

 

                "Demanda_Predicha" 

 

            ] = ( 

 

                predicciones 

 

            ) 

 

 

            st.success( 

                "Predicciones generadas correctamente." 

            ) 

 

 

            st.subheader( 

                "Resultados" 

            ) 

 

 

            st.dataframe( 

                resultado 

            ) 

 

 

            st.metric( 

 

                "Demanda promedio predicha", 

 

                round( 

 

                    resultado[ 

                        "Demanda_Predicha" 

                    ].mean(), 

 

                    2 

 

                ) 

 

            ) 

 

 

            csv = ( 

 

                resultado 

 

                .to_csv( 

                    index=False 

                ) 

 

                .encode( 

                    "utf-8" 

                ) 

 

            ) 

 

 

            st.download_button( 

 

                "Descargar predicciones", 

 

                data=csv, 

 

                file_name="predicciones_demanda.csv", 

 

                mime="text/csv" 

 

            ) 

 

 

        except Exception as error: 

 

            st.error( 

                "No fue posible realizar la predicción." 

            ) 

 

            st.write( 

                error 

            ) 

 

 

st.markdown( 

    """ 

### Interpretación empresarial 

 

La predicción representa una estimación de la 

demanda esperada para el siguiente periodo. 

 

El resultado debe utilizarse como apoyo para 

decisiones de inventario, planificación, 

abastecimiento y comercialización. 

 

No debe considerarse una predicción infalible. 

""" 

) 

