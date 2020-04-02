import turtle as t


def tree(dis, time, decayrate=0.6, lang=40, rang=50):
    if time != 0:
        t.forward(dis)
        t.left(lang)
        tree(decayrate * dis, time - 1, decayrate, lang, rang)
        t.right(lang + rang)
        tree(decayrate * dis, time - 1, decayrate, lang, rang)
        t.left(rang)
        t.backward(dis)


if __name__ == "__main__":
    t.screensize(1600, 1200)
    t.pencolor('red')
    t.speed(10)
    t.goto(0, -100)
    t.left(90)
    tree(100, 10, 0.7, 35, 35)

