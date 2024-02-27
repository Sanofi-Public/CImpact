import snowflake.connector
import pandas
import yaml


class Connection:
    """class to connect to snowflake

        Please tweak config.yaml file before utilization
    ...

    Attributes
    ----------
    con :
        connection to snowflake

    Methods
    -------

    """

    connections: dict

    con: snowflake.connector.SnowflakeConnection

    def __init__(self, zone: str):
        """
        Parameters
        ----------
        zone : str
            the zone to which you need to connect

        """
        self.connect(zone)

    def connect(self, zone: str):
        """return a connector to snowflake database
        Parameters
        ----------
        zone : str
            the zone to which you need to connect

        Raises
        ------
        Exception if yaml format is incorrect

        """

        # getting config
        with open("config.yaml", "r") as file:
            config = yaml.safe_load(file)

        self.connections = config["connections"]

        # creating connection to snowflake

        # check login
        if "login" in config and len(config["login"]) > 0:
            # check connections completeness

            if all(
                k in self.connections[zone]
                for k in ["account", "warehouse", "database", "role"]
            ):
                self.con = snowflake.connector.connect(
                    user=config["login"],
                    account=self.connections[zone]["account"],
                    authenticator="externalbrowser",
                    warehouse=self.connections[zone]["warehouse"],
                    database=self.connections[zone]["database"],
                    role=self.connections[zone]["role"],
                )
            else:
                raise Exception("Incorrect yaml file - missing connections info")
        else:
            # exception if login info are not there
            raise Exception("Incorrect yaml file - no login")

    def fetch(self, sql):
        """
        function to query sql

         Parameters
        ----------
        sql : str
            a string representing a query

        Returns
        -------
        a pandas dataframe with the data fetched


        """

        return self.con.cursor().execute(sql).fetch_pandas_all()
