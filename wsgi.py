from packages import dash_app

if __name__ == '__main__':
    #from waitress import serve
    #serve(dash_app.server, host="0.0.0.0", port=8080)
    dash_app.run_server(debug=False)
