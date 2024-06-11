from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, lit, unix_timestamp, lead, lag, when, udf
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, StringType, FloatType, BooleanType
from geopy.distance import geodesic
import os
import matplotlib.pyplot as plt

# Sukuriame Spark sesiją
spark = SparkSession.builder \
    .appName("ClosestVesselsAnalysis") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .config("spark.network.timeout", "600s") \
    .getOrCreate()

# Apibrėžiame duomenų schemą
schema = StructType([
    StructField("Timestamp", StringType(), True),
    StructField("Type", StringType(), True),
    StructField("MMSI", StringType(), True),
    StructField("Latitude", FloatType(), True),
    StructField("Longitude", FloatType(), True)
])

# Nurodykite direktoriją, kurioje yra failai
directory = 'C:/Users/User/OneDrive - Vilnius University/Desktop/bigdata/aisdk-2021-12/'

# Skaitymas failų
file_list = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".csv") and "aisdk-2021-12" in f]

# Sukuriame tuščią DataFrame, į kurį sujungsime visus failus
combined_data = spark.createDataFrame([], schema)

# Skaitymo ir apdorojimo ciklas
for file_path in file_list:
    print(f"Processing file: {file_path}")
    data = spark.read.csv(file_path, schema=schema, header=True)
    combined_data = combined_data.union(data)

# Talpiname tarpinius duomenis atmintyje
combined_data.cache()

# Filtruojame neteisingas platumos ir ilgumos reikšmes
combined_data = combined_data.filter((col("Latitude") >= -90.0) & (col("Latitude") <= 90.0))
combined_data = combined_data.filter((col("Longitude") >= -180.0) & (col("Longitude") <= 180.0))

# Konvertuojame Timestamp į tinkamą formatą
combined_data = combined_data.withColumn("Timestamp", to_timestamp(col("Timestamp"), "dd/MM/yyyy HH:mm:ss"))

# Filtruojame duomenis pagal datą
start_date = "2021-12-01 00:00:00"
end_date = "2021-12-31 23:59:59"
combined_data = combined_data.filter((col("Timestamp") >= lit(start_date)) & 
                                     (col("Timestamp") <= lit(end_date)))

# Apibrėžiame funkciją patikrinti, ar laivas yra nurodytame spindulyje
def is_within_radius(lat, lon, center_lat=55.225000, center_lon=14.245000, radius_km=50):
    return geodesic((lat, lon), (center_lat, center_lon)).km <= radius_km

# Filtruojame duomenis tiesiogiai naudojant funkciją
is_within_radius_udf = udf(is_within_radius, BooleanType())
filtered_data = combined_data.filter(is_within_radius_udf(col("Latitude"), col("Longitude")))

# Patikriname, kiek eilučių liko po filtravimo
filtered_count = filtered_data.count()
print(f"Rows after filtering by radius: {filtered_count}")

if filtered_count > 0:
    # Rikiuojame duomenis pagal MMSI ir Timestamp
    windowSpec = Window.partitionBy("MMSI").orderBy("Timestamp")
    filtered_data = filtered_data.withColumn("PrevLatitude", lag("Latitude").over(windowSpec)) \
                                 .withColumn("PrevLongitude", lag("Longitude").over(windowSpec)) \
                                 .withColumn("PrevTimestamp", lag("Timestamp").over(windowSpec))

    # Apskaičiuojame atstumą tarp sekų
    def geodesic_distance(lat1, lon1, lat2, lon2):
        if None in (lat1, lon1, lat2, lon2):
            return None
        return geodesic((lat1, lon1), (lat2, lon2)).km

    geodesic_udf = udf(geodesic_distance, FloatType())
    filtered_data = filtered_data.withColumn("Distance", geodesic_udf(col("Latitude"), col("Longitude"), col("PrevLatitude"), col("PrevLongitude")))

    # Apskaičiuojame laiką tarp sekų
    filtered_data = filtered_data.withColumn("TimeDiff", (unix_timestamp(col("Timestamp")) - unix_timestamp(col("PrevTimestamp"))))

    # Apskaičiuojame greitį
    filtered_data = filtered_data.withColumn("Speed", when(col("Distance").isNotNull() & (col("TimeDiff") > 0), (col("Distance") / (col("TimeDiff") / 3600))).otherwise(None))

    # Filtruojame triukšmą - pašaliname nerealius greičius (> 107 km/h)
    # Laivai negali plaukti greičiau, nei 107 km/h (https://en.wikipedia.org/wiki/HSC_Francisco)
    filtered_data = filtered_data.filter((col("Speed") <= 107) & (col("Speed").isNotNull()))

    # Patikrinkime, kiek eilučių turi apskaičiuotą atstumą ir greitį
    distance_count = filtered_data.filter(col("Distance").isNotNull()).count()
    print(f"Rows with calculated distance and realistic speed: {distance_count}")

    if distance_count > 0:
        # Filtruojame artimiausius laivus
        windowSpecLead = Window.orderBy("Distance")
        filtered_data = filtered_data.withColumn("NextMMSI", lead("MMSI").over(windowSpecLead)) \
                                     .withColumn("NextLatitude", lead("Latitude").over(windowSpecLead)) \
                                     .withColumn("NextLongitude", lead("Longitude").over(windowSpecLead)) \
                                     .withColumn("NextTimestamp", lead("Timestamp").over(windowSpecLead))

        filtered_data = filtered_data.withColumn("NextDistance", geodesic_udf(col("Latitude"), col("Longitude"), col("NextLatitude"), col("NextLongitude")))

        closest_vessels = filtered_data.orderBy("NextDistance").limit(1).collect()

        # Vizualizuojame trajektoriją
        if closest_vessels and closest_vessels[0]["NextMMSI"]:
            mmsi1, mmsi2 = closest_vessels[0]["MMSI"], closest_vessels[0]["NextMMSI"]
            print(f"Found closest vessels: {mmsi1} and {mmsi2}")

            traj1 = filtered_data.filter((col("MMSI") == mmsi1) & 
                                         (unix_timestamp(col("Timestamp")) >= unix_timestamp(lit(closest_vessels[0]["Timestamp"])) - 600) & 
                                         (unix_timestamp(col("Timestamp")) <= unix_timestamp(lit(closest_vessels[0]["Timestamp"])) + 600)).toPandas()

            traj2 = filtered_data.filter((col("MMSI") == mmsi2) & 
                                         (unix_timestamp(col("Timestamp")) >= unix_timestamp(lit(closest_vessels[0]["NextTimestamp"])) - 600) & 
                                         (unix_timestamp(col("Timestamp")) <= unix_timestamp(lit(closest_vessels[0]["NextTimestamp"])) + 600)).toPandas()

            plt.figure(figsize=(10, 6))
            plt.plot(traj1['Longitude'], traj1['Latitude'], marker='o', label=f'Laivas 1: {mmsi1}')
            plt.plot(traj2['Longitude'], traj2['Latitude'], marker='o', label=f'Laivas 2: {mmsi2}')
            plt.xlabel('Ilguma')
            plt.ylabel('Platuma')
            plt.title('Laivų trajektorija 10 minučių prieš ir po susitikimo momento')
            plt.legend()
            plt.grid(True)
            plt.show()
        else:
            print("Nerasta jokių artimų laivų.")
    else:
        print("Nerasta jokių laivų su apskaičiuotu atstumu ir realistišku greičiu.")
else:
    print("Nerasta jokių laivų nurodytame spindulyje.")

# Sustabdome Spark sesiją
spark.stop()


